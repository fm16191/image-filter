default: all

all:
	g++ -I${HOME}/softs/FreeImage/include modif_img.cpp -L${HOME}/softs/FreeImage/lib/ -lfreeimage -o modif_img.exe

clean:
	rm -f *.o modif_img.exe
