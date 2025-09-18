# 📌 IPCV251 Image Super Resolution (Phương pháp truyền thống, không Deep Learning)

**Created:** 9/18/2025, 8:13:08 AM  

---

## 🎯 Tổng quan  
Dự án này thực hiện các phương pháp super-resolution truyền thống (không dùng mô hình học sâu) theo ba nhóm chính:

- Nội suy (Interpolation-Based)  
- Tái xây dựng (Reconstruction-Based)  
- Dựa vào Machine Learning truyền thống (Patch / Example-Based)  

**Deliverables:**

- Repository GitHub với mã + notebook Colab  
- Demo Gradio (chạy trong Colab, “Run All” ổn)  
- Báo cáo Overleaf (PDF) mỗi người viết phần riêng + hợp nhất  

---

## 👥 Phân công nhiệm vụ + Keywords / Tài liệu tham khảo

### 🔹 Task 1 – Phương pháp Nội suy (Interpolation-Based) — 1 người Coding:

**Coding:**

- Triển khai các phương pháp: Nearest, Bilinear, Bicubic  
- Thêm 1 phương pháp nâng cao: Edge-Directed / Spline / Wavelet  

**Report:**

- Lý thuyết về interpolation (toán + trực giác)  
- Ưu / nhược điểm (mờ, blockiness, đường biên, texture)  
- Thí nghiệm: so sánh ảnh đầu vào LR → SR → ảnh gốc (Ground Truth)  

**Deliverables:**

- Hàm python: `sr_interpolation(img, method)`  
- Hình ảnh / biểu đồ so sánh các phương pháp  
- Phần báo cáo  

**Keywords / Papers / Project Tham khảo:**

- https://github.com/OpenDEM/DCCI4DEM?tab=readme-ov-file  
- https://github.com/Kirstihly/Edge-Directed_Interpolation  
- https://github.com/czxrrr/super-resolution (Matlab)  
- https://github.com/tecnickcom/inedi (Matlab)  
- https://github.com/KaiGuo-Vision/MultiscaleInterpolation (Matlab)  
- https://opendem.info/dcci.html  
- *New Edge-Directed Interpolation* — X. Li & M. T. Orchard, IEEE Trans. Image Process., 2001  
- *Performance Evaluation of Edge-Directed Interpolation Methods for Images* — Yu, Zhu, Wu, Xie, arXiv 2013  
- *Structure Tensor Based Image Interpolation Method* (Baghaie, Z. Yu, 2014)  

---

### 🔹 Task 2 – Phương pháp Phản hồi / Tái xây dựng (Reconstruction-Based) — 2 người Coding:

**Coding:**

- Người 1: Iterative Back-Projection  
- Người 2: Lọc Wiener + Regularization (ví dụ Tikhonov, MAP, hoặc các ràng buộc smooth / edge)  

**Report:**

- Mô hình suy giảm (degradation model: blur, downsampling, noise)  
- Thuật toán: iterative refinement, Wiener filter, cách regularization  
- Hiển thị các bước lặp, phân tích hội tụ / trade-off (thời gian vs chất lượng)  

**Deliverables:**

- Hàm python: `sr_backprojection(img)`, `sr_wiener(img)`  
- Hình ảnh minh hoạ bước lặp / ảnh kết quả so sánh  
- Báo cáo  

**Keywords / Papers / Project Tham khảo:**

- https://github.com/czxrrr/super-resolution (Matlab)  
- *Image Super-Resolution - Iterative Back Projection Algorithm* (Matlab File Exchange)  
- *Iterative Back Projection based Image Resolution Enhancement* — Pejman Rasti, Hasan Demirel, Gholamreza Anbarjafari  
- *Multi-example feature-constrained back-projection method for image super-resolution* (Zhang, Gai, Xin, Xuemei Li, 2016)  

---

### 🔹 Task 3 – Machine Learning truyền thống (Patch / Example-Based) — 2 người Coding:

**Coding:**

- Người 1: Xây dựng dictionary / thu thập patch ví dụ  
- Người 2: Neighbor embedding / tái tạo patch + blend các patch  

**Report:**

- Giới thiệu khái niệm Example-based / Patch-based SR  
- Toán / thuật toán: cách tìm patch giống nhau, embedding, blending  
- Thí nghiệm: kết quả, artifacts thường gặp (block, mismatch), so sánh với interpolation & reconstruction  

**Deliverables:**

- Hàm python: `sr_patchbased(img)`  
- Demo sử dụng benchmark nhỏ (ví dụ: Set5 / Set14)  
- Báo cáo + slides cho cả hai  

**Keywords / Papers / Project Tham khảo:**

- https://github.com/jesse1029/example-based-super-resolution (Matlab)  
- *Multi-example feature-constrained back-projection method for image super-resolution* — link.springer.com (2016)  
- Kỹ thuật neighbor embedding super-resolution  
- Paper *“Seven ways to improve example-based single image super resolution”* (Timofte et al.) — https://arxiv.org/pdf/1511.02228  

---

## 🔹 Cuối cùng làm chung

- Đánh giá (Evaluation): Viết hàm PSNR + SSIM, tổng hợp kết quả các phương pháp vào bảng  
- Demo Gradio (trong Colab): Cho phép upload ảnh → chọn phương pháp → hiển thị ảnh SR + metrics  
- Báo cáo:  
  - Phần mở đầu (Leader)  
  - Các phần phương pháp (mỗi nhóm viết phần mình)  
  - Phần đánh giá + thảo luận (Team)  
  - Kết luận (Leader / phối hợp chỉnh sửa)  

---

## ✅ Deliverables cuối cùng

- Repository GitHub (có cấu trúc rõ ràng: `/src, /notebooks, /report`)  
- Notebook Colab: đảm bảo “Run All” chạy được  
- Báo cáo PDF
