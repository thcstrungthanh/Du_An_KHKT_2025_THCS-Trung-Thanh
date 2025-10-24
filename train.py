import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc  # Dọn bộ nhớ
from torch.cuda.amp import autocast, GradScaler  # Tối ưu bộ nhớ

# Cấu hình seed để kết quả ổn định
def dat_hat_giong(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Chuẩn bị dữ liệu với cấu trúc train/val
def tao_du_lieu(thu_muc_goc, thu_muc_dich, ti_le_train=0.8):
    # Tạo cấu trúc thư mục train/val
    phan_loai = ['train', 'val']
    cac_benh = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']
    for phan in phan_loai:
        for benh in cac_benh:
            os.makedirs(os.path.join(thu_muc_dich, phan, benh), exist_ok=True)
    
    # Chia dữ liệu theo từng loại bệnh
    for benh in cac_benh:
        duong_dan_nguon = os.path.join(thu_muc_goc, benh)
        if not os.path.exists(duong_dan_nguon):
            print(f"Cảnh báo: {duong_dan_nguon} không tồn tại")
            continue
            
        # Lấy danh sách ảnh
        ds_anh = [f for f in os.listdir(duong_dan_nguon) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Chia train/val
        anh_train, anh_val = train_test_split(ds_anh, train_size=ti_le_train, random_state=42)
        
        # Sao chép ảnh vào thư mục đích
        for anh in anh_train:
            nguon = os.path.join(duong_dan_nguon, anh)
            dich = os.path.join(thu_muc_dich, 'train', benh, anh)
            shutil.copy2(nguon, dich)
            
        for anh in anh_val:
            nguon = os.path.join(duong_dan_nguon, anh)
            dich = os.path.join(thu_muc_dich, 'val', benh, anh)
            shutil.copy2(nguon, dich)
            
        print(f"{benh}: {len(anh_train)} ảnh train, {len(anh_val)} ảnh validation")

# Hàm biến đổi ảnh (augment + chuẩn hóa)
def lay_bien_doi():
    bien_doi_train = transforms.Compose([
        transforms.Resize((160, 160)),  
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    bien_doi_val = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return bien_doi_train, bien_doi_val

# Huấn luyện 1 epoch
def train_epoch(model, dl_train, ham_mat, toi_uu, thiet_bi, chia_ty_le):
    model.train()
    tong_mat = 0.0
    tong_dung = 0
    tong_mau = 0
    
    thanh_tien_trinh = tqdm(dl_train, desc='Huấn luyện')
    for anh, nhan in thanh_tien_trinh:
        anh = anh.to(thiet_bi)
        nhan = nhan.to(thiet_bi)
        
        toi_uu.zero_grad(set_to_none=True)
        
        # Sử dụng mixed precision để tiết kiệm bộ nhớ
        with autocast():
            dau_ra = model(anh)
            _, du_doan = torch.max(dau_ra, 1)
            mat = ham_mat(dau_ra, nhan)
        
        chia_ty_le.scale(mat).backward()
        chia_ty_le.step(toi_uu)
        chia_ty_le.update()
        
        tong_mat += mat.item() * anh.size(0)
        tong_dung += torch.sum(du_doan == nhan.data)
        tong_mau += anh.size(0)
        
        thanh_tien_trinh.set_postfix({'mat': mat.item(), 'do_chinh_xac': (tong_dung.double()/tong_mau).item()})
        
        # Dọn bộ nhớ
        del dau_ra, mat
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
    
    mat_epoch = tong_mat / tong_mau
    do_chinh_xac_epoch = tong_dung.double() / tong_mau
    
    return mat_epoch, do_chinh_xac_epoch

# Đánh giá mô hình
def danh_gia(model, dl_val, ham_mat, thiet_bi):
    model.eval()
    tong_mat = 0.0
    tong_dung = 0
    tong_mau = 0
    
    with torch.no_grad():
        for anh, nhan in tqdm(dl_val, desc='Đánh giá'):
            anh = anh.to(thiet_bi)
            nhan = nhan.to(thiet_bi)
            
            dau_ra = model(anh)
            _, du_doan = torch.max(dau_ra, 1)
            mat = ham_mat(dau_ra, nhan)
            
            tong_mat += mat.item() * anh.size(0)
            tong_dung += torch.sum(du_doan == nhan.data)
            tong_mau += anh.size(0)
    
    mat_epoch = tong_mat / tong_mau
    do_chinh_xac_epoch = tong_dung.double() / tong_mau
    
    return mat_epoch, do_chinh_xac_epoch

def main():
    # Cấu hình
    dat_hat_giong(42)
    thiet_bi = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {thiet_bi}")
    
    # Đường dẫn dữ liệu
    thu_muc_goc = "Du_lieu_Anh_Cay_Mia"
    thu_muc_dich = "du_lieu_train"
    
    # Chuẩn bị dữ liệu
    print("Đang chuẩn bị dữ liệu...")
    tao_du_lieu(thu_muc_goc, thu_muc_dich)
    
    # Biến đổi ảnh
    bien_doi_train, bien_doi_val = lay_bien_doi()
    
    # Tải dữ liệu
    du_lieu_train = datasets.ImageFolder(
        os.path.join(thu_muc_dich, 'train'),
        transform=bien_doi_train
    )
    du_lieu_val = datasets.ImageFolder(
        os.path.join(thu_muc_dich, 'val'),
        transform=bien_doi_val
    )
    
    # Tạo dataloader
    dl_train = DataLoader(
        du_lieu_train, 
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    dl_val = DataLoader(
        du_lieu_val, 
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Khởi tạo model
    print("Đang khởi tạo mô hình...")
    model = models.resnet18(pretrained=True)
    so_dac_trung = model.fc.in_features
    model.fc = nn.Linear(so_dac_trung, len(du_lieu_train.classes))
    model = model.to(thiet_bi)
    
    # Hàm mất mát, bộ tối ưu và scheduler
    ham_mat = nn.CrossEntropyLoss()
    toi_uu = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    dieu_chinh_lr = optim.lr_scheduler.ReduceLROnPlateau(
        toi_uu, 
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Sử dụng mixed precision
    chia_ty_le = GradScaler()
    
    # Huấn luyện
    print("Bắt đầu huấn luyện mô hình...")
    so_epoch = 20
    do_chinh_xac_tot_nhat = 0.0
    
    try:
        for epoch in range(so_epoch):
            print(f'\nEpoch {epoch+1}/{so_epoch}')
            
            # Huấn luyện
            mat_train, acc_train = train_epoch(model, dl_train, ham_mat, toi_uu, thiet_bi, chia_ty_le)
            print(f'Train - Mất mát: {mat_train:.4f} | Độ chính xác: {acc_train:.4f}')
            
            # Đánh giá
            mat_val, acc_val = danh_gia(model, dl_val, ham_mat, thiet_bi)
            print(f'Val - Mất mát: {mat_val:.4f} | Độ chính xác: {acc_val:.4f}')
            
            # Điều chỉnh learning rate
            dieu_chinh_lr.step(acc_val)
            
            # Lưu mô hình tốt nhất
            if acc_val > do_chinh_xac_tot_nhat:
                do_chinh_xac_tot_nhat = acc_val
                model_script = torch.jit.script(model)
                os.makedirs('Mo_Hinh', exist_ok=True)
                model_script.save('Mo_Hinh/Mo_hinh_du_doan_benh.pth')
                print(f"Đã lưu mô hình với độ chính xác: {do_chinh_xac_tot_nhat:.4f}")
            
            # Dọn bộ nhớ
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    except KeyboardInterrupt:
        print("\nHuấn luyện bị dừng bởi người dùng!")
        if do_chinh_xac_tot_nhat > 0:
            torch.jit.script(model).save('Mo_Hinh/Mo_hinh_du_doan_benh_loi.pth')
    
    print("\n✅ Huấn luyện hoàn tất!")
    print(f"🎯 Độ chính xác tốt nhất trên tập validation: {do_chinh_xac_tot_nhat:.4f}")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    main()
