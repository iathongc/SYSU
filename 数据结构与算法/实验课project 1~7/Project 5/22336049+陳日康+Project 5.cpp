#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include "CImg.h"
using namespace std;
using namespace cimg_library;
#pragma pack(1)

struct Pixel {
    int x;
    int y;
    int red;
    int green;
    int blue;
};

vector<Pixel> convertToTriplets(const CImg<unsigned char>& image) {
    vector<Pixel> triplets;

    cimg_forXY(image, x, y) {
        Pixel pixel = {x, y, image(x, y, 0, 0), image(x, y, 0, 1), image(x, y, 0, 2)};
        triplets.push_back(pixel);
    }

    return triplets;
}

CImg<unsigned char> convertToGrayscale(const CImg<unsigned char>& image) {
    return image.get_RGBtoYCbCr().channel(0);
}

CImg<unsigned char> resizeImage(const CImg<unsigned char>& image, int newWidth, int newHeight) {
    return image.get_resize(newWidth, newHeight, 1, 1);
}

void compressAndSave(const CImg<unsigned char>& image, const string& binFileName) {
    // 打開二進位檔案進行寫入
    ofstream outFile(binFileName, ios::binary);

    if (outFile.is_open()) {
        // 寫入圖像的寬度和高度
        const unsigned int width = image.width();
        const unsigned int height = image.height();
        outFile.write(reinterpret_cast<const char*>(&width), sizeof(width));
        outFile.write(reinterpret_cast<const char*>(&height), sizeof(height));

        // 逐個圖元寫入
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                Pixel pixel = {x, y, image(x, y, 0, 0), image(x, y, 0, 1), image(x, y, 0, 2)};
                
                // 逐個成員變數寫入
                outFile.write(reinterpret_cast<const char*>(&pixel.x), sizeof(pixel.x));
                outFile.write(reinterpret_cast<const char*>(&pixel.y), sizeof(pixel.y));
                outFile.write(reinterpret_cast<const char*>(&pixel.red), sizeof(pixel.red));
                outFile.write(reinterpret_cast<const char*>(&pixel.green), sizeof(pixel.green));
                outFile.write(reinterpret_cast<const char*>(&pixel.blue), sizeof(pixel.blue));
            }
        }

        cout << "壓縮完成。" << endl;

        // 關閉文件
        outFile.close();
    }
	else {
        cout << "Error: Could not open file for writing." << endl;
    }
}

void readAndDecompress(const string& binFileName, const string& outputImageName) {
    cout << "從文件讀取：" << binFileName << endl;
    // 打開二進位檔案進行讀取
    ifstream inFile(binFileName, ios::binary);

    if (!inFile) {
        cerr << "Error: Could not open file for reading: " << binFileName << endl;
        return;
    }

    // 讀取圖像的寬度和高度
    unsigned int width, height;
    inFile.read(reinterpret_cast<char*>(&width), sizeof(width));
    inFile.read(reinterpret_cast<char*>(&height), sizeof(height));

    // 創建用於解壓縮的圖像
    CImg<unsigned char> decompressedImage(width, height, 1, 3, 0);

    // 逐個圖元讀取
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            Pixel pixel;
            
            // 逐個成員變數讀取
            inFile.read(reinterpret_cast<char*>(&pixel.x), sizeof(pixel.x));
            inFile.read(reinterpret_cast<char*>(&pixel.y), sizeof(pixel.y));
            inFile.read(reinterpret_cast<char*>(&pixel.red), sizeof(pixel.red));
            inFile.read(reinterpret_cast<char*>(&pixel.green), sizeof(pixel.green));
            inFile.read(reinterpret_cast<char*>(&pixel.blue), sizeof(pixel.blue));

            // 填充解壓後的圖像
            decompressedImage(pixel.x, pixel.y, 0, 0) = pixel.red;
            decompressedImage(pixel.x, pixel.y, 0, 1) = pixel.green;
            decompressedImage(pixel.x, pixel.y, 0, 2) = pixel.blue;
        }
    }

    // 保存解壓後的圖像
    decompressedImage.save(outputImageName.c_str());

    // 關閉文件
    inFile.close();
}

void displayGrayscale(const CImg<unsigned char>& originalImage) {
    CImg<unsigned char> grayscaleImage = convertToGrayscale(originalImage);

    CImgDisplay grayscaleDisplay(grayscaleImage, "GrayPic");
    while (!grayscaleDisplay.is_closed()) {
        grayscaleDisplay.wait();
    }
    cout << "灰度圖已顯示並保存為新圖像！" << endl;
    string outputFileName = "grayscale-image.ppm";
    grayscaleImage.save(outputFileName.c_str());
    cout << "圖像已成功保存為 " << outputFileName << "！" << endl;
    cout << endl;
}

void resizeAndDisplay(const CImg<unsigned char>& originalImage, int newWidth, int newHeight) {
    CImg<unsigned char> resizedImage = resizeImage(originalImage, newWidth, newHeight);

    CImgDisplay resizedDisplay(resizedImage, "ResizedPic");
    while (!resizedDisplay.is_closed()) {
        resizedDisplay.wait();
    }

    cout << "圖像已調整大小並保存為新圖像！" << endl;
    string outputFileName = "resized-image.ppm";
    resizedImage.save(outputFileName.c_str());
    cout << "圖像已成功保存為 " << outputFileName << "！" << endl;
    cout << endl;
}

int main() {
    const string binFileName = "compressed_image.bin";
	const string outputImageName = "decompressed_image.ppm";
    cout << setw(25) << "簡單圖像處理程序" << endl;
    cout << endl;

    CImg<unsigned char> selectedImage; // 用於保存當前選擇的圖像

    while (true) {
        // 載入圖像
        CImg<unsigned char> colorBlockImage("color-block.ppm");
        CImg<unsigned char> lena128Image("lena-128-gray.ppm");
        CImg<unsigned char> lena512Image("lena-512-gray.ppm");

        // 顯示圖像選項
    	cout << "================================================" << endl;
        cout << endl;
        cout << "請選擇要處理的圖像" << endl;
        cout << "a: color-block.ppm" << endl;
        cout << "b: lena-128-gray.ppm" << endl;
        cout << "c: lena-512-gray.ppm" << endl;
        cout << "請輸入圖像編號 (a, b, c)：";

        char imageChoice;
        cin >> imageChoice;
        cout << endl;
		cout << "已顯示圖像！（請關閉圖像視窗以便繼續操作）" << endl;

        switch (imageChoice) {
            case 'a':
                selectedImage = colorBlockImage;
                break;
            case 'b':
                selectedImage = lena128Image;
                break;
            case 'c':
                selectedImage = lena512Image;
                break;
            default:
                cout << "輸入有誤，請選擇有效的圖像編號！" << endl;
                return 1;
        }
		cout << "Selected image dimensions: " << selectedImage.width() << " x " << selectedImage.height() << endl;
        // 顯示選定圖像
        CImgDisplay originalDisplay(selectedImage, "OriginPic");
        while (!originalDisplay.is_closed()) {
        	originalDisplay.wait();
        }
        
		cout << endl;

		do {
        	// 顯示影像處理選項
    		cout << "================================================" << endl;
    		cout << endl;
        	cout << "影像處理選項：" << endl;
        	cout << "1: 壓縮並保存" << endl;
        	cout << "2: 讀取解壓檔並保存" << endl;        
        	cout << "3: 轉為灰度圖像並顯示和保存" << endl;
        	cout << "4: 調整圖像大小並保存" << endl;
       		cout << "0: 退出程式" << endl;

        	int functionChoice;		
			switch (imageChoice) {
				case 'a':
					cout << "請輸入功能選擇 (1, 2, 3)：";
					break;
				case 'b':
					cout << "請輸入功能選擇 (1, 2, 4)：";		    	
					selectedImage = lena128Image;
					break;
				case 'c':
					cout << "請輸入功能選擇 (1, 2, 4)：";
					selectedImage = lena512Image;
					break;
				default:
					cout << "輸入有誤，請選擇有效的編號！" << endl;
					return 1;
			}
        	cin >> functionChoice;        
        	cout << endl;

			switch (functionChoice) {
            	case 1: {
    				// 壓縮並保存圖像
    				compressAndSave(selectedImage, binFileName);
                	cout << "圖像已成功壓縮保存為二進位檔案！" << endl;
    				break;
            	}
            	case 2: {
					// 讀取並解壓縮圖像
                	readAndDecompress(binFileName, outputImageName);
                	cout << "二進位檔案已成功解壓縮為圖像並保存！" << endl;
                	break;
            	}
            	case 3: {
                    if (imageChoice != 'a') {
                        cout << "選擇錯誤，請重新選擇！" << endl;
                        continue;
                    }
                	displayGrayscale(selectedImage);
                	break;
            	}
            	case 4: {
                    if (imageChoice == 'a') {
                        cout << "選擇錯誤，請重新選擇！" << endl;
                        continue;
                    }
                	int newWidth, newHeight;
                	cout << "請輸入調整後的寬度和高度：";
                	cin >> newWidth >> newHeight;
                	resizeAndDisplay(selectedImage, newWidth, newHeight);
                	break;
            	}
            	case 0:
                	cout << "程式已結束。" << endl;
                	return 0;
            	default:
                	cout << "無效的選擇，請重新輸入。" << endl;
                	return 1;
        	}
        	// 詢問使用者是否返回選擇圖像
			char returnChoice;
			cout << "是否繼續選擇當前圖像繼續操作 (y/n)？";
			cin >> returnChoice;
			cout << endl;
			if (returnChoice != 'y') {
		    	break;
			}
    	}
		while (true);
	}
	return 0;
}
