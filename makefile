all: chessboard digits
chessboard: chessboard.cpp
	g++ -Wall -Werror -Wextra -ggdb3 -ffast-math -fopenmp chessboard.cpp -o chessboard
digits: digits.cpp
	g++ -Wall -Werror -Wextra -ggdb3 -ffast-math -fopenmp digits.cpp -o digits
