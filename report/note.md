

# 1. Rouge
* Tập các metric dùng để đánh giá mô hình tóm tắt văn bản hoặc dịch máy (bằng cách so sánh output của mô hình với đoạn tóm tắt/dịch mẫu, sử dụng độ đo precision, recall, f1 để so sánh mức độ overlap giữa output mô hình và output mẫu)
* Rouge-N (Rouge-1, Rouge-2,...): xét các n-gram của output mô hình và output mẫu
* Rouge-L: xét LCS (chuỗi con chung dài nhất, chú 1 chuỗi x là chuỗi con của chuỗi y thì không nhất thiết các phần tử của x phải là liên tiếp trong y, có thể là các phần tử nằm ở các vị trí cách nhau trong y, nhưng phải đúng thứ tự xuất hiện)
	* Hiểu được cách tính khi chỉ xét 2 cặp câu
	* Xây dựng nên công thức cho tập câu (đọc paper)
* Rouge-S,...