CXX=g++
CXXFLAGS=-std=c++11

OPENCV = -I/usr/local/include/opencv -I/usr/local/include -L/usr/local/lib -lopencv_videostab -lopencv_stitching -lopencv_superres -lopencv_tracking -lopencv_reg -lopencv_datasets -lopencv_text -lopencv_stereo -lopencv_aruco -lopencv_xobjdetect -lopencv_plot -lopencv_freetype -lopencv_ccalib -lopencv_xfeatures2d -lopencv_shape -lopencv_ml -lopencv_dpm -lopencv_xphoto -lopencv_face -lopencv_photo -lopencv_objdetect -lopencv_bioinspired -lopencv_rgbd -lopencv_structured_light -lopencv_fuzzy -lopencv_img_hash -lopencv_bgsegm -lopencv_dnn_objdetect -lopencv_dnn -lopencv_optflow -lopencv_video -lopencv_hfs -lopencv_line_descriptor -lopencv_ximgproc -lopencv_calib3d -lopencv_surface_matching -lopencv_phase_unwrapping -lopencv_saliency -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_flann -lopencv_core

main:
	g++ -std=c++11 main.cpp `pkg-config --cflags --libs opencv`
	
dunno:
	$(CXX) $(CXXFLAGS) $@.cpp $(OPENCV)
