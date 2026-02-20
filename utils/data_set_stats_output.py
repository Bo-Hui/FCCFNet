import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
import argparse


class NUDT_SIRST_Dataset(Dataset):
    """
    NUDT-SIRST数据集类，用于计算统计量
    """
    def __init__(self, root_dir, transform=None):
        """
        参数:
            root_dir: 数据集根目录
            transform: 图像转换操作
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        
        # 获取图像文件名列表
        self.image_names = self._get_image_names()
        
        # 默认转换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform
    
    def _get_image_names(self):
        """获取图像文件名列表"""
        image_names = []
        # 遍历masks文件夹获取文件名（因为masks文件夹在目录结构中可见）
        for filename in os.listdir(self.mask_dir):
            if filename.endswith('.png'):
                # 去除扩展名
                image_name = filename[:-4]
                image_names.append(image_name)
        return image_names
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.image_names)
    
    def __getitem__(self, idx):
        """获取指定索引的图像和掩码"""
        image_name = self.image_names[idx]
        
        # 构建图像和掩码的完整路径
        image_path = os.path.join(self.image_dir, image_name + '.png')
        mask_path = os.path.join(self.mask_dir, image_name + '.png')
        
        # 读取图像和掩码
        img = Image.open(image_path).convert('RGB')  # 转换为RGB
        mask = Image.open(mask_path)  # 掩码通常是单通道
        
        # 应用转换
        img = self.transform(img)
        mask = transforms.ToTensor()(mask)
        
        return img, mask

class IRSTD1K_Dataset(Dataset):
    """
    IRSTD-1k数据集类，用于计算统计量
    
    参数:
        root_dir: 数据集根目录
        args: 兼容原有接口的参数对象（可选）
        transform: 图像转换操作（可选）
    """
    def __init__(self, root_dir=None, args=None, transform=None):
        # 优先使用root_dir参数，如果没有则使用args或默认路径
        if root_dir is not None:
            self.root_dir = root_dir
        elif args is not None and hasattr(args, 'data_path'):
            self.root_dir = args.data_path
        else:
            self.root_dir = 'datasets/IRSTD-1k'
            
        self.image_dir = os.path.join(self.root_dir, 'images')
        self.mask_dir = os.path.join(self.root_dir, 'masks')
        
        # 获取图像文件名列表
        self.image_names = self._get_image_names()
        
        # 默认转换 - 仅使用ToTensor()，不使用Normalize（用于统计量计算）
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform
    
    def _get_image_names(self):
        """获取图像文件名列表"""
        image_names = []
        # 遍历masks文件夹获取文件名
        for filename in os.listdir(self.mask_dir):
            if filename.endswith('.png'):
                # 去除扩展名
                image_name = filename[:-4]
                image_names.append(image_name)
        return image_names
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.image_names)
    
    def __getitem__(self, idx):
        """获取指定索引的图像和掩码"""
        image_name = self.image_names[idx]
        
        # 构建图像和掩码的完整路径
        image_path = os.path.join(self.image_dir, image_name + '.png')
        mask_path = os.path.join(self.mask_dir, image_name + '.png')
        
        # 读取图像和掩码
        img = Image.open(image_path).convert('RGB')  # 转换为RGB
        mask = Image.open(mask_path)  # 掩码通常是单通道
        
        # 应用转换
        img = self.transform(img)
        mask = transforms.ToTensor()(mask)
        
        return img, mask

def get_mean_std(data_set):
    loader = DataLoader(data_set, batch_size=16, num_workers=1, shuffle=False, pin_memory=True)

    sum_of_pixels = torch.zeros(3)
    sum_of_square_error = torch.zeros(3)
    
    # 动态计算像素数量
    total_pixels = 0
    for X, _ in loader:
        batch_size, channels, height, width = X.shape
        total_pixels += batch_size * height * width
        for d in range(3):
            sum_of_pixels[d] += X[:, d, :, :].sum()
    
    _mean = sum_of_pixels / total_pixels
    
    # 计算标准差
    for X, _ in loader:
        for d in range(3):
            sum_of_square_error[d] += ((X[:, d, :, :] - _mean[d]).pow(2)).sum()
    
    _std = torch.sqrt(sum_of_square_error / total_pixels)

    return list(_mean.numpy()), list(_std.numpy())


# 使用示例
if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='计算数据集的均值和标准差')
    parser.add_argument('--dataset', type=str, default='IRSTD-1k',
                       choices=['NUDT-SIRST', 'IRSTD-1k'],
                       help='要计算的数据集类型')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='数据集的路径（如果不指定，将使用默认路径）')
    args = parser.parse_args()
    
    # 根据选择的数据集类型处理路径
    if args.dataset_path is None:
        if args.dataset == 'NUDT-SIRST':
            args.dataset_path = 'd:/Work/project/experimenting/红外/分割/ilnet/ILNet_新实验_进行中_mask更改/datasets/NUDT-SIRST'
        elif args.dataset == 'IRSTD-1k':
            args.dataset_path = 'd:/Work/project/experimenting/红外/分割/ilnet/ILNet_新实验_进行中_mask更改/datasets/IRSTD-1k'
    
    print(f'正在加载{args.dataset}数据集...')
    print(f'数据集路径: {args.dataset_path}')
    
    # 创建数据集对象
    if args.dataset == 'NUDT-SIRST':
        dataset = NUDT_SIRST_Dataset(args.dataset_path)
    elif args.dataset == 'IRSTD-1k':
        dataset = IRSTD1K_Dataset(args.dataset_path)
    
    print(f'数据集大小: {len(dataset)} 张图像')
    
    # 计算均值和标准差
    print(f'正在计算均值和标准差...')
    mean, std = get_mean_std(dataset)
    
    print(f'计算完成!')
    print(f'均值: {mean}')
    print(f'标准差: {std}')
    
    # 格式化输出，方便复制到代码中
    print(f'\n可直接用于transforms.Normalize的参数:')
    print(f'mean=[{mean[0]:.8f}, {mean[1]:.8f}, {mean[2]:.8f}]')
    print(f'std=[{std[0]:.8f}, {std[1]:.8f}, {std[2]:.8f}]')
