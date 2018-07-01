from scipy.optimize import root, fsolve
import numpy as np

def gen(x):
    def f1(p):
        return np.array([x[0][0]*p[0]+x[0][1]*p[1]+x[0][2]*p[2]+x[0][3]*p[3]+x[0][4]*p[4]+x[0][5]*p[5]+x[0][6]*p[6]+x[0][7]*p[7]+x[0][8]*p[8]+x[0][9]*p[9]+x[0][10]*p[10]+x[0][11]*p[11]+x[0][12]*p[12]+x[0][13]*p[13]+x[0][14]*p[14]+x[0][15]*p[15]+x[0][16]*p[16]+x[0][17]*p[17]+x[0][18]*p[18]+x[0][19]*p[19]+x[0][20]*p[20]+x[0][21]*p[21]+x[0][22]*p[22]+x[0][23]*p[23]+x[0][24]*p[24]+x[0][25]*p[25]+x[0][26]*p[26]+x[0][27]*p[27]+x[0][28]*p[28]+x[0][29]*p[29]+x[0][30]*p[30]+x[0][31]*p[31]+x[0][32]*p[32]+x[0][33]*p[33]+x[0][34]*p[34]+x[0][35]*p[35]+x[0][36]*p[36]+x[0][37]*p[37]-1/p[0],x[1][0]*p[0]+x[1][1]*p[1]+x[1][2]*p[2]+x[1][3]*p[3]+x[1][4]*p[4]+x[1][5]*p[5]+x[1][6]*p[6]+x[1][7]*p[7]+x[1][8]*p[8]+x[1][9]*p[9]+x[1][10]*p[10]+x[1][11]*p[11]+x[1][12]*p[12]+x[1][13]*p[13]+x[1][14]*p[14]+x[1][15]*p[15]+x[1][16]*p[16]+x[1][17]*p[17]+x[1][18]*p[18]+x[1][19]*p[19]+x[1][20]*p[20]+x[1][21]*p[21]+x[1][22]*p[22]+x[1][23]*p[23]+x[1][24]*p[24]+x[1][25]*p[25]+x[1][26]*p[26]+x[1][27]*p[27]+x[1][28]*p[28]+x[1][29]*p[29]+x[1][30]*p[30]+x[1][31]*p[31]+x[1][32]*p[32]+x[1][33]*p[33]+x[1][34]*p[34]+x[1][35]*p[35]+x[1][36]*p[36]+x[1][37]*p[37]-1/p[1],x[2][0]*p[0]+x[2][1]*p[1]+x[2][2]*p[2]+x[2][3]*p[3]+x[2][4]*p[4]+x[2][5]*p[5]+x[2][6]*p[6]+x[2][7]*p[7]+x[2][8]*p[8]+x[2][9]*p[9]+x[2][10]*p[10]+x[2][11]*p[11]+x[2][12]*p[12]+x[2][13]*p[13]+x[2][14]*p[14]+x[2][15]*p[15]+x[2][16]*p[16]+x[2][17]*p[17]+x[2][18]*p[18]+x[2][19]*p[19]+x[2][20]*p[20]+x[2][21]*p[21]+x[2][22]*p[22]+x[2][23]*p[23]+x[2][24]*p[24]+x[2][25]*p[25]+x[2][26]*p[26]+x[2][27]*p[27]+x[2][28]*p[28]+x[2][29]*p[29]+x[2][30]*p[30]+x[2][31]*p[31]+x[2][32]*p[32]+x[2][33]*p[33]+x[2][34]*p[34]+x[2][35]*p[35]+x[2][36]*p[36]+x[2][37]*p[37]-1/p[2],x[3][0]*p[0]+x[3][1]*p[1]+x[3][2]*p[2]+x[3][3]*p[3]+x[3][4]*p[4]+x[3][5]*p[5]+x[3][6]*p[6]+x[3][7]*p[7]+x[3][8]*p[8]+x[3][9]*p[9]+x[3][10]*p[10]+x[3][11]*p[11]+x[3][12]*p[12]+x[3][13]*p[13]+x[3][14]*p[14]+x[3][15]*p[15]+x[3][16]*p[16]+x[3][17]*p[17]+x[3][18]*p[18]+x[3][19]*p[19]+x[3][20]*p[20]+x[3][21]*p[21]+x[3][22]*p[22]+x[3][23]*p[23]+x[3][24]*p[24]+x[3][25]*p[25]+x[3][26]*p[26]+x[3][27]*p[27]+x[3][28]*p[28]+x[3][29]*p[29]+x[3][30]*p[30]+x[3][31]*p[31]+x[3][32]*p[32]+x[3][33]*p[33]+x[3][34]*p[34]+x[3][35]*p[35]+x[3][36]*p[36]+x[3][37]*p[37]-1/p[3],x[4][0]*p[0]+x[4][1]*p[1]+x[4][2]*p[2]+x[4][3]*p[3]+x[4][4]*p[4]+x[4][5]*p[5]+x[4][6]*p[6]+x[4][7]*p[7]+x[4][8]*p[8]+x[4][9]*p[9]+x[4][10]*p[10]+x[4][11]*p[11]+x[4][12]*p[12]+x[4][13]*p[13]+x[4][14]*p[14]+x[4][15]*p[15]+x[4][16]*p[16]+x[4][17]*p[17]+x[4][18]*p[18]+x[4][19]*p[19]+x[4][20]*p[20]+x[4][21]*p[21]+x[4][22]*p[22]+x[4][23]*p[23]+x[4][24]*p[24]+x[4][25]*p[25]+x[4][26]*p[26]+x[4][27]*p[27]+x[4][28]*p[28]+x[4][29]*p[29]+x[4][30]*p[30]+x[4][31]*p[31]+x[4][32]*p[32]+x[4][33]*p[33]+x[4][34]*p[34]+x[4][35]*p[35]+x[4][36]*p[36]+x[4][37]*p[37]-1/p[4],x[5][0]*p[0]+x[5][1]*p[1]+x[5][2]*p[2]+x[5][3]*p[3]+x[5][4]*p[4]+x[5][5]*p[5]+x[5][6]*p[6]+x[5][7]*p[7]+x[5][8]*p[8]+x[5][9]*p[9]+x[5][10]*p[10]+x[5][11]*p[11]+x[5][12]*p[12]+x[5][13]*p[13]+x[5][14]*p[14]+x[5][15]*p[15]+x[5][16]*p[16]+x[5][17]*p[17]+x[5][18]*p[18]+x[5][19]*p[19]+x[5][20]*p[20]+x[5][21]*p[21]+x[5][22]*p[22]+x[5][23]*p[23]+x[5][24]*p[24]+x[5][25]*p[25]+x[5][26]*p[26]+x[5][27]*p[27]+x[5][28]*p[28]+x[5][29]*p[29]+x[5][30]*p[30]+x[5][31]*p[31]+x[5][32]*p[32]+x[5][33]*p[33]+x[5][34]*p[34]+x[5][35]*p[35]+x[5][36]*p[36]+x[5][37]*p[37]-1/p[5],x[6][0]*p[0]+x[6][1]*p[1]+x[6][2]*p[2]+x[6][3]*p[3]+x[6][4]*p[4]+x[6][5]*p[5]+x[6][6]*p[6]+x[6][7]*p[7]+x[6][8]*p[8]+x[6][9]*p[9]+x[6][10]*p[10]+x[6][11]*p[11]+x[6][12]*p[12]+x[6][13]*p[13]+x[6][14]*p[14]+x[6][15]*p[15]+x[6][16]*p[16]+x[6][17]*p[17]+x[6][18]*p[18]+x[6][19]*p[19]+x[6][20]*p[20]+x[6][21]*p[21]+x[6][22]*p[22]+x[6][23]*p[23]+x[6][24]*p[24]+x[6][25]*p[25]+x[6][26]*p[26]+x[6][27]*p[27]+x[6][28]*p[28]+x[6][29]*p[29]+x[6][30]*p[30]+x[6][31]*p[31]+x[6][32]*p[32]+x[6][33]*p[33]+x[6][34]*p[34]+x[6][35]*p[35]+x[6][36]*p[36]+x[6][37]*p[37]-1/p[6],x[7][0]*p[0]+x[7][1]*p[1]+x[7][2]*p[2]+x[7][3]*p[3]+x[7][4]*p[4]+x[7][5]*p[5]+x[7][6]*p[6]+x[7][7]*p[7]+x[7][8]*p[8]+x[7][9]*p[9]+x[7][10]*p[10]+x[7][11]*p[11]+x[7][12]*p[12]+x[7][13]*p[13]+x[7][14]*p[14]+x[7][15]*p[15]+x[7][16]*p[16]+x[7][17]*p[17]+x[7][18]*p[18]+x[7][19]*p[19]+x[7][20]*p[20]+x[7][21]*p[21]+x[7][22]*p[22]+x[7][23]*p[23]+x[7][24]*p[24]+x[7][25]*p[25]+x[7][26]*p[26]+x[7][27]*p[27]+x[7][28]*p[28]+x[7][29]*p[29]+x[7][30]*p[30]+x[7][31]*p[31]+x[7][32]*p[32]+x[7][33]*p[33]+x[7][34]*p[34]+x[7][35]*p[35]+x[7][36]*p[36]+x[7][37]*p[37]-1/p[7],x[8][0]*p[0]+x[8][1]*p[1]+x[8][2]*p[2]+x[8][3]*p[3]+x[8][4]*p[4]+x[8][5]*p[5]+x[8][6]*p[6]+x[8][7]*p[7]+x[8][8]*p[8]+x[8][9]*p[9]+x[8][10]*p[10]+x[8][11]*p[11]+x[8][12]*p[12]+x[8][13]*p[13]+x[8][14]*p[14]+x[8][15]*p[15]+x[8][16]*p[16]+x[8][17]*p[17]+x[8][18]*p[18]+x[8][19]*p[19]+x[8][20]*p[20]+x[8][21]*p[21]+x[8][22]*p[22]+x[8][23]*p[23]+x[8][24]*p[24]+x[8][25]*p[25]+x[8][26]*p[26]+x[8][27]*p[27]+x[8][28]*p[28]+x[8][29]*p[29]+x[8][30]*p[30]+x[8][31]*p[31]+x[8][32]*p[32]+x[8][33]*p[33]+x[8][34]*p[34]+x[8][35]*p[35]+x[8][36]*p[36]+x[8][37]*p[37]-1/p[8],x[9][0]*p[0]+x[9][1]*p[1]+x[9][2]*p[2]+x[9][3]*p[3]+x[9][4]*p[4]+x[9][5]*p[5]+x[9][6]*p[6]+x[9][7]*p[7]+x[9][8]*p[8]+x[9][9]*p[9]+x[9][10]*p[10]+x[9][11]*p[11]+x[9][12]*p[12]+x[9][13]*p[13]+x[9][14]*p[14]+x[9][15]*p[15]+x[9][16]*p[16]+x[9][17]*p[17]+x[9][18]*p[18]+x[9][19]*p[19]+x[9][20]*p[20]+x[9][21]*p[21]+x[9][22]*p[22]+x[9][23]*p[23]+x[9][24]*p[24]+x[9][25]*p[25]+x[9][26]*p[26]+x[9][27]*p[27]+x[9][28]*p[28]+x[9][29]*p[29]+x[9][30]*p[30]+x[9][31]*p[31]+x[9][32]*p[32]+x[9][33]*p[33]+x[9][34]*p[34]+x[9][35]*p[35]+x[9][36]*p[36]+x[9][37]*p[37]-1/p[9],x[10][0]*p[0]+x[10][1]*p[1]+x[10][2]*p[2]+x[10][3]*p[3]+x[10][4]*p[4]+x[10][5]*p[5]+x[10][6]*p[6]+x[10][7]*p[7]+x[10][8]*p[8]+x[10][9]*p[9]+x[10][10]*p[10]+x[10][11]*p[11]+x[10][12]*p[12]+x[10][13]*p[13]+x[10][14]*p[14]+x[10][15]*p[15]+x[10][16]*p[16]+x[10][17]*p[17]+x[10][18]*p[18]+x[10][19]*p[19]+x[10][20]*p[20]+x[10][21]*p[21]+x[10][22]*p[22]+x[10][23]*p[23]+x[10][24]*p[24]+x[10][25]*p[25]+x[10][26]*p[26]+x[10][27]*p[27]+x[10][28]*p[28]+x[10][29]*p[29]+x[10][30]*p[30]+x[10][31]*p[31]+x[10][32]*p[32]+x[10][33]*p[33]+x[10][34]*p[34]+x[10][35]*p[35]+x[10][36]*p[36]+x[10][37]*p[37]-1/p[10],x[11][0]*p[0]+x[11][1]*p[1]+x[11][2]*p[2]+x[11][3]*p[3]+x[11][4]*p[4]+x[11][5]*p[5]+x[11][6]*p[6]+x[11][7]*p[7]+x[11][8]*p[8]+x[11][9]*p[9]+x[11][10]*p[10]+x[11][11]*p[11]+x[11][12]*p[12]+x[11][13]*p[13]+x[11][14]*p[14]+x[11][15]*p[15]+x[11][16]*p[16]+x[11][17]*p[17]+x[11][18]*p[18]+x[11][19]*p[19]+x[11][20]*p[20]+x[11][21]*p[21]+x[11][22]*p[22]+x[11][23]*p[23]+x[11][24]*p[24]+x[11][25]*p[25]+x[11][26]*p[26]+x[11][27]*p[27]+x[11][28]*p[28]+x[11][29]*p[29]+x[11][30]*p[30]+x[11][31]*p[31]+x[11][32]*p[32]+x[11][33]*p[33]+x[11][34]*p[34]+x[11][35]*p[35]+x[11][36]*p[36]+x[11][37]*p[37]-1/p[11],x[12][0]*p[0]+x[12][1]*p[1]+x[12][2]*p[2]+x[12][3]*p[3]+x[12][4]*p[4]+x[12][5]*p[5]+x[12][6]*p[6]+x[12][7]*p[7]+x[12][8]*p[8]+x[12][9]*p[9]+x[12][10]*p[10]+x[12][11]*p[11]+x[12][12]*p[12]+x[12][13]*p[13]+x[12][14]*p[14]+x[12][15]*p[15]+x[12][16]*p[16]+x[12][17]*p[17]+x[12][18]*p[18]+x[12][19]*p[19]+x[12][20]*p[20]+x[12][21]*p[21]+x[12][22]*p[22]+x[12][23]*p[23]+x[12][24]*p[24]+x[12][25]*p[25]+x[12][26]*p[26]+x[12][27]*p[27]+x[12][28]*p[28]+x[12][29]*p[29]+x[12][30]*p[30]+x[12][31]*p[31]+x[12][32]*p[32]+x[12][33]*p[33]+x[12][34]*p[34]+x[12][35]*p[35]+x[12][36]*p[36]+x[12][37]*p[37]-1/p[12],x[13][0]*p[0]+x[13][1]*p[1]+x[13][2]*p[2]+x[13][3]*p[3]+x[13][4]*p[4]+x[13][5]*p[5]+x[13][6]*p[6]+x[13][7]*p[7]+x[13][8]*p[8]+x[13][9]*p[9]+x[13][10]*p[10]+x[13][11]*p[11]+x[13][12]*p[12]+x[13][13]*p[13]+x[13][14]*p[14]+x[13][15]*p[15]+x[13][16]*p[16]+x[13][17]*p[17]+x[13][18]*p[18]+x[13][19]*p[19]+x[13][20]*p[20]+x[13][21]*p[21]+x[13][22]*p[22]+x[13][23]*p[23]+x[13][24]*p[24]+x[13][25]*p[25]+x[13][26]*p[26]+x[13][27]*p[27]+x[13][28]*p[28]+x[13][29]*p[29]+x[13][30]*p[30]+x[13][31]*p[31]+x[13][32]*p[32]+x[13][33]*p[33]+x[13][34]*p[34]+x[13][35]*p[35]+x[13][36]*p[36]+x[13][37]*p[37]-1/p[13],x[14][0]*p[0]+x[14][1]*p[1]+x[14][2]*p[2]+x[14][3]*p[3]+x[14][4]*p[4]+x[14][5]*p[5]+x[14][6]*p[6]+x[14][7]*p[7]+x[14][8]*p[8]+x[14][9]*p[9]+x[14][10]*p[10]+x[14][11]*p[11]+x[14][12]*p[12]+x[14][13]*p[13]+x[14][14]*p[14]+x[14][15]*p[15]+x[14][16]*p[16]+x[14][17]*p[17]+x[14][18]*p[18]+x[14][19]*p[19]+x[14][20]*p[20]+x[14][21]*p[21]+x[14][22]*p[22]+x[14][23]*p[23]+x[14][24]*p[24]+x[14][25]*p[25]+x[14][26]*p[26]+x[14][27]*p[27]+x[14][28]*p[28]+x[14][29]*p[29]+x[14][30]*p[30]+x[14][31]*p[31]+x[14][32]*p[32]+x[14][33]*p[33]+x[14][34]*p[34]+x[14][35]*p[35]+x[14][36]*p[36]+x[14][37]*p[37]-1/p[14],x[15][0]*p[0]+x[15][1]*p[1]+x[15][2]*p[2]+x[15][3]*p[3]+x[15][4]*p[4]+x[15][5]*p[5]+x[15][6]*p[6]+x[15][7]*p[7]+x[15][8]*p[8]+x[15][9]*p[9]+x[15][10]*p[10]+x[15][11]*p[11]+x[15][12]*p[12]+x[15][13]*p[13]+x[15][14]*p[14]+x[15][15]*p[15]+x[15][16]*p[16]+x[15][17]*p[17]+x[15][18]*p[18]+x[15][19]*p[19]+x[15][20]*p[20]+x[15][21]*p[21]+x[15][22]*p[22]+x[15][23]*p[23]+x[15][24]*p[24]+x[15][25]*p[25]+x[15][26]*p[26]+x[15][27]*p[27]+x[15][28]*p[28]+x[15][29]*p[29]+x[15][30]*p[30]+x[15][31]*p[31]+x[15][32]*p[32]+x[15][33]*p[33]+x[15][34]*p[34]+x[15][35]*p[35]+x[15][36]*p[36]+x[15][37]*p[37]-1/p[15],x[16][0]*p[0]+x[16][1]*p[1]+x[16][2]*p[2]+x[16][3]*p[3]+x[16][4]*p[4]+x[16][5]*p[5]+x[16][6]*p[6]+x[16][7]*p[7]+x[16][8]*p[8]+x[16][9]*p[9]+x[16][10]*p[10]+x[16][11]*p[11]+x[16][12]*p[12]+x[16][13]*p[13]+x[16][14]*p[14]+x[16][15]*p[15]+x[16][16]*p[16]+x[16][17]*p[17]+x[16][18]*p[18]+x[16][19]*p[19]+x[16][20]*p[20]+x[16][21]*p[21]+x[16][22]*p[22]+x[16][23]*p[23]+x[16][24]*p[24]+x[16][25]*p[25]+x[16][26]*p[26]+x[16][27]*p[27]+x[16][28]*p[28]+x[16][29]*p[29]+x[16][30]*p[30]+x[16][31]*p[31]+x[16][32]*p[32]+x[16][33]*p[33]+x[16][34]*p[34]+x[16][35]*p[35]+x[16][36]*p[36]+x[16][37]*p[37]-1/p[16],x[17][0]*p[0]+x[17][1]*p[1]+x[17][2]*p[2]+x[17][3]*p[3]+x[17][4]*p[4]+x[17][5]*p[5]+x[17][6]*p[6]+x[17][7]*p[7]+x[17][8]*p[8]+x[17][9]*p[9]+x[17][10]*p[10]+x[17][11]*p[11]+x[17][12]*p[12]+x[17][13]*p[13]+x[17][14]*p[14]+x[17][15]*p[15]+x[17][16]*p[16]+x[17][17]*p[17]+x[17][18]*p[18]+x[17][19]*p[19]+x[17][20]*p[20]+x[17][21]*p[21]+x[17][22]*p[22]+x[17][23]*p[23]+x[17][24]*p[24]+x[17][25]*p[25]+x[17][26]*p[26]+x[17][27]*p[27]+x[17][28]*p[28]+x[17][29]*p[29]+x[17][30]*p[30]+x[17][31]*p[31]+x[17][32]*p[32]+x[17][33]*p[33]+x[17][34]*p[34]+x[17][35]*p[35]+x[17][36]*p[36]+x[17][37]*p[37]-1/p[17],x[18][0]*p[0]+x[18][1]*p[1]+x[18][2]*p[2]+x[18][3]*p[3]+x[18][4]*p[4]+x[18][5]*p[5]+x[18][6]*p[6]+x[18][7]*p[7]+x[18][8]*p[8]+x[18][9]*p[9]+x[18][10]*p[10]+x[18][11]*p[11]+x[18][12]*p[12]+x[18][13]*p[13]+x[18][14]*p[14]+x[18][15]*p[15]+x[18][16]*p[16]+x[18][17]*p[17]+x[18][18]*p[18]+x[18][19]*p[19]+x[18][20]*p[20]+x[18][21]*p[21]+x[18][22]*p[22]+x[18][23]*p[23]+x[18][24]*p[24]+x[18][25]*p[25]+x[18][26]*p[26]+x[18][27]*p[27]+x[18][28]*p[28]+x[18][29]*p[29]+x[18][30]*p[30]+x[18][31]*p[31]+x[18][32]*p[32]+x[18][33]*p[33]+x[18][34]*p[34]+x[18][35]*p[35]+x[18][36]*p[36]+x[18][37]*p[37]-1/p[18],x[19][0]*p[0]+x[19][1]*p[1]+x[19][2]*p[2]+x[19][3]*p[3]+x[19][4]*p[4]+x[19][5]*p[5]+x[19][6]*p[6]+x[19][7]*p[7]+x[19][8]*p[8]+x[19][9]*p[9]+x[19][10]*p[10]+x[19][11]*p[11]+x[19][12]*p[12]+x[19][13]*p[13]+x[19][14]*p[14]+x[19][15]*p[15]+x[19][16]*p[16]+x[19][17]*p[17]+x[19][18]*p[18]+x[19][19]*p[19]+x[19][20]*p[20]+x[19][21]*p[21]+x[19][22]*p[22]+x[19][23]*p[23]+x[19][24]*p[24]+x[19][25]*p[25]+x[19][26]*p[26]+x[19][27]*p[27]+x[19][28]*p[28]+x[19][29]*p[29]+x[19][30]*p[30]+x[19][31]*p[31]+x[19][32]*p[32]+x[19][33]*p[33]+x[19][34]*p[34]+x[19][35]*p[35]+x[19][36]*p[36]+x[19][37]*p[37]-1/p[19],x[20][0]*p[0]+x[20][1]*p[1]+x[20][2]*p[2]+x[20][3]*p[3]+x[20][4]*p[4]+x[20][5]*p[5]+x[20][6]*p[6]+x[20][7]*p[7]+x[20][8]*p[8]+x[20][9]*p[9]+x[20][10]*p[10]+x[20][11]*p[11]+x[20][12]*p[12]+x[20][13]*p[13]+x[20][14]*p[14]+x[20][15]*p[15]+x[20][16]*p[16]+x[20][17]*p[17]+x[20][18]*p[18]+x[20][19]*p[19]+x[20][20]*p[20]+x[20][21]*p[21]+x[20][22]*p[22]+x[20][23]*p[23]+x[20][24]*p[24]+x[20][25]*p[25]+x[20][26]*p[26]+x[20][27]*p[27]+x[20][28]*p[28]+x[20][29]*p[29]+x[20][30]*p[30]+x[20][31]*p[31]+x[20][32]*p[32]+x[20][33]*p[33]+x[20][34]*p[34]+x[20][35]*p[35]+x[20][36]*p[36]+x[20][37]*p[37]-1/p[20],x[21][0]*p[0]+x[21][1]*p[1]+x[21][2]*p[2]+x[21][3]*p[3]+x[21][4]*p[4]+x[21][5]*p[5]+x[21][6]*p[6]+x[21][7]*p[7]+x[21][8]*p[8]+x[21][9]*p[9]+x[21][10]*p[10]+x[21][11]*p[11]+x[21][12]*p[12]+x[21][13]*p[13]+x[21][14]*p[14]+x[21][15]*p[15]+x[21][16]*p[16]+x[21][17]*p[17]+x[21][18]*p[18]+x[21][19]*p[19]+x[21][20]*p[20]+x[21][21]*p[21]+x[21][22]*p[22]+x[21][23]*p[23]+x[21][24]*p[24]+x[21][25]*p[25]+x[21][26]*p[26]+x[21][27]*p[27]+x[21][28]*p[28]+x[21][29]*p[29]+x[21][30]*p[30]+x[21][31]*p[31]+x[21][32]*p[32]+x[21][33]*p[33]+x[21][34]*p[34]+x[21][35]*p[35]+x[21][36]*p[36]+x[21][37]*p[37]-1/p[21],x[22][0]*p[0]+x[22][1]*p[1]+x[22][2]*p[2]+x[22][3]*p[3]+x[22][4]*p[4]+x[22][5]*p[5]+x[22][6]*p[6]+x[22][7]*p[7]+x[22][8]*p[8]+x[22][9]*p[9]+x[22][10]*p[10]+x[22][11]*p[11]+x[22][12]*p[12]+x[22][13]*p[13]+x[22][14]*p[14]+x[22][15]*p[15]+x[22][16]*p[16]+x[22][17]*p[17]+x[22][18]*p[18]+x[22][19]*p[19]+x[22][20]*p[20]+x[22][21]*p[21]+x[22][22]*p[22]+x[22][23]*p[23]+x[22][24]*p[24]+x[22][25]*p[25]+x[22][26]*p[26]+x[22][27]*p[27]+x[22][28]*p[28]+x[22][29]*p[29]+x[22][30]*p[30]+x[22][31]*p[31]+x[22][32]*p[32]+x[22][33]*p[33]+x[22][34]*p[34]+x[22][35]*p[35]+x[22][36]*p[36]+x[22][37]*p[37]-1/p[22],x[23][0]*p[0]+x[23][1]*p[1]+x[23][2]*p[2]+x[23][3]*p[3]+x[23][4]*p[4]+x[23][5]*p[5]+x[23][6]*p[6]+x[23][7]*p[7]+x[23][8]*p[8]+x[23][9]*p[9]+x[23][10]*p[10]+x[23][11]*p[11]+x[23][12]*p[12]+x[23][13]*p[13]+x[23][14]*p[14]+x[23][15]*p[15]+x[23][16]*p[16]+x[23][17]*p[17]+x[23][18]*p[18]+x[23][19]*p[19]+x[23][20]*p[20]+x[23][21]*p[21]+x[23][22]*p[22]+x[23][23]*p[23]+x[23][24]*p[24]+x[23][25]*p[25]+x[23][26]*p[26]+x[23][27]*p[27]+x[23][28]*p[28]+x[23][29]*p[29]+x[23][30]*p[30]+x[23][31]*p[31]+x[23][32]*p[32]+x[23][33]*p[33]+x[23][34]*p[34]+x[23][35]*p[35]+x[23][36]*p[36]+x[23][37]*p[37]-1/p[23],x[24][0]*p[0]+x[24][1]*p[1]+x[24][2]*p[2]+x[24][3]*p[3]+x[24][4]*p[4]+x[24][5]*p[5]+x[24][6]*p[6]+x[24][7]*p[7]+x[24][8]*p[8]+x[24][9]*p[9]+x[24][10]*p[10]+x[24][11]*p[11]+x[24][12]*p[12]+x[24][13]*p[13]+x[24][14]*p[14]+x[24][15]*p[15]+x[24][16]*p[16]+x[24][17]*p[17]+x[24][18]*p[18]+x[24][19]*p[19]+x[24][20]*p[20]+x[24][21]*p[21]+x[24][22]*p[22]+x[24][23]*p[23]+x[24][24]*p[24]+x[24][25]*p[25]+x[24][26]*p[26]+x[24][27]*p[27]+x[24][28]*p[28]+x[24][29]*p[29]+x[24][30]*p[30]+x[24][31]*p[31]+x[24][32]*p[32]+x[24][33]*p[33]+x[24][34]*p[34]+x[24][35]*p[35]+x[24][36]*p[36]+x[24][37]*p[37]-1/p[24],x[25][0]*p[0]+x[25][1]*p[1]+x[25][2]*p[2]+x[25][3]*p[3]+x[25][4]*p[4]+x[25][5]*p[5]+x[25][6]*p[6]+x[25][7]*p[7]+x[25][8]*p[8]+x[25][9]*p[9]+x[25][10]*p[10]+x[25][11]*p[11]+x[25][12]*p[12]+x[25][13]*p[13]+x[25][14]*p[14]+x[25][15]*p[15]+x[25][16]*p[16]+x[25][17]*p[17]+x[25][18]*p[18]+x[25][19]*p[19]+x[25][20]*p[20]+x[25][21]*p[21]+x[25][22]*p[22]+x[25][23]*p[23]+x[25][24]*p[24]+x[25][25]*p[25]+x[25][26]*p[26]+x[25][27]*p[27]+x[25][28]*p[28]+x[25][29]*p[29]+x[25][30]*p[30]+x[25][31]*p[31]+x[25][32]*p[32]+x[25][33]*p[33]+x[25][34]*p[34]+x[25][35]*p[35]+x[25][36]*p[36]+x[25][37]*p[37]-1/p[25],x[26][0]*p[0]+x[26][1]*p[1]+x[26][2]*p[2]+x[26][3]*p[3]+x[26][4]*p[4]+x[26][5]*p[5]+x[26][6]*p[6]+x[26][7]*p[7]+x[26][8]*p[8]+x[26][9]*p[9]+x[26][10]*p[10]+x[26][11]*p[11]+x[26][12]*p[12]+x[26][13]*p[13]+x[26][14]*p[14]+x[26][15]*p[15]+x[26][16]*p[16]+x[26][17]*p[17]+x[26][18]*p[18]+x[26][19]*p[19]+x[26][20]*p[20]+x[26][21]*p[21]+x[26][22]*p[22]+x[26][23]*p[23]+x[26][24]*p[24]+x[26][25]*p[25]+x[26][26]*p[26]+x[26][27]*p[27]+x[26][28]*p[28]+x[26][29]*p[29]+x[26][30]*p[30]+x[26][31]*p[31]+x[26][32]*p[32]+x[26][33]*p[33]+x[26][34]*p[34]+x[26][35]*p[35]+x[26][36]*p[36]+x[26][37]*p[37]-1/p[26],x[27][0]*p[0]+x[27][1]*p[1]+x[27][2]*p[2]+x[27][3]*p[3]+x[27][4]*p[4]+x[27][5]*p[5]+x[27][6]*p[6]+x[27][7]*p[7]+x[27][8]*p[8]+x[27][9]*p[9]+x[27][10]*p[10]+x[27][11]*p[11]+x[27][12]*p[12]+x[27][13]*p[13]+x[27][14]*p[14]+x[27][15]*p[15]+x[27][16]*p[16]+x[27][17]*p[17]+x[27][18]*p[18]+x[27][19]*p[19]+x[27][20]*p[20]+x[27][21]*p[21]+x[27][22]*p[22]+x[27][23]*p[23]+x[27][24]*p[24]+x[27][25]*p[25]+x[27][26]*p[26]+x[27][27]*p[27]+x[27][28]*p[28]+x[27][29]*p[29]+x[27][30]*p[30]+x[27][31]*p[31]+x[27][32]*p[32]+x[27][33]*p[33]+x[27][34]*p[34]+x[27][35]*p[35]+x[27][36]*p[36]+x[27][37]*p[37]-1/p[27],x[28][0]*p[0]+x[28][1]*p[1]+x[28][2]*p[2]+x[28][3]*p[3]+x[28][4]*p[4]+x[28][5]*p[5]+x[28][6]*p[6]+x[28][7]*p[7]+x[28][8]*p[8]+x[28][9]*p[9]+x[28][10]*p[10]+x[28][11]*p[11]+x[28][12]*p[12]+x[28][13]*p[13]+x[28][14]*p[14]+x[28][15]*p[15]+x[28][16]*p[16]+x[28][17]*p[17]+x[28][18]*p[18]+x[28][19]*p[19]+x[28][20]*p[20]+x[28][21]*p[21]+x[28][22]*p[22]+x[28][23]*p[23]+x[28][24]*p[24]+x[28][25]*p[25]+x[28][26]*p[26]+x[28][27]*p[27]+x[28][28]*p[28]+x[28][29]*p[29]+x[28][30]*p[30]+x[28][31]*p[31]+x[28][32]*p[32]+x[28][33]*p[33]+x[28][34]*p[34]+x[28][35]*p[35]+x[28][36]*p[36]+x[28][37]*p[37]-1/p[28],x[29][0]*p[0]+x[29][1]*p[1]+x[29][2]*p[2]+x[29][3]*p[3]+x[29][4]*p[4]+x[29][5]*p[5]+x[29][6]*p[6]+x[29][7]*p[7]+x[29][8]*p[8]+x[29][9]*p[9]+x[29][10]*p[10]+x[29][11]*p[11]+x[29][12]*p[12]+x[29][13]*p[13]+x[29][14]*p[14]+x[29][15]*p[15]+x[29][16]*p[16]+x[29][17]*p[17]+x[29][18]*p[18]+x[29][19]*p[19]+x[29][20]*p[20]+x[29][21]*p[21]+x[29][22]*p[22]+x[29][23]*p[23]+x[29][24]*p[24]+x[29][25]*p[25]+x[29][26]*p[26]+x[29][27]*p[27]+x[29][28]*p[28]+x[29][29]*p[29]+x[29][30]*p[30]+x[29][31]*p[31]+x[29][32]*p[32]+x[29][33]*p[33]+x[29][34]*p[34]+x[29][35]*p[35]+x[29][36]*p[36]+x[29][37]*p[37]-1/p[29],x[30][0]*p[0]+x[30][1]*p[1]+x[30][2]*p[2]+x[30][3]*p[3]+x[30][4]*p[4]+x[30][5]*p[5]+x[30][6]*p[6]+x[30][7]*p[7]+x[30][8]*p[8]+x[30][9]*p[9]+x[30][10]*p[10]+x[30][11]*p[11]+x[30][12]*p[12]+x[30][13]*p[13]+x[30][14]*p[14]+x[30][15]*p[15]+x[30][16]*p[16]+x[30][17]*p[17]+x[30][18]*p[18]+x[30][19]*p[19]+x[30][20]*p[20]+x[30][21]*p[21]+x[30][22]*p[22]+x[30][23]*p[23]+x[30][24]*p[24]+x[30][25]*p[25]+x[30][26]*p[26]+x[30][27]*p[27]+x[30][28]*p[28]+x[30][29]*p[29]+x[30][30]*p[30]+x[30][31]*p[31]+x[30][32]*p[32]+x[30][33]*p[33]+x[30][34]*p[34]+x[30][35]*p[35]+x[30][36]*p[36]+x[30][37]*p[37]-1/p[30],x[31][0]*p[0]+x[31][1]*p[1]+x[31][2]*p[2]+x[31][3]*p[3]+x[31][4]*p[4]+x[31][5]*p[5]+x[31][6]*p[6]+x[31][7]*p[7]+x[31][8]*p[8]+x[31][9]*p[9]+x[31][10]*p[10]+x[31][11]*p[11]+x[31][12]*p[12]+x[31][13]*p[13]+x[31][14]*p[14]+x[31][15]*p[15]+x[31][16]*p[16]+x[31][17]*p[17]+x[31][18]*p[18]+x[31][19]*p[19]+x[31][20]*p[20]+x[31][21]*p[21]+x[31][22]*p[22]+x[31][23]*p[23]+x[31][24]*p[24]+x[31][25]*p[25]+x[31][26]*p[26]+x[31][27]*p[27]+x[31][28]*p[28]+x[31][29]*p[29]+x[31][30]*p[30]+x[31][31]*p[31]+x[31][32]*p[32]+x[31][33]*p[33]+x[31][34]*p[34]+x[31][35]*p[35]+x[31][36]*p[36]+x[31][37]*p[37]-1/p[31],x[32][0]*p[0]+x[32][1]*p[1]+x[32][2]*p[2]+x[32][3]*p[3]+x[32][4]*p[4]+x[32][5]*p[5]+x[32][6]*p[6]+x[32][7]*p[7]+x[32][8]*p[8]+x[32][9]*p[9]+x[32][10]*p[10]+x[32][11]*p[11]+x[32][12]*p[12]+x[32][13]*p[13]+x[32][14]*p[14]+x[32][15]*p[15]+x[32][16]*p[16]+x[32][17]*p[17]+x[32][18]*p[18]+x[32][19]*p[19]+x[32][20]*p[20]+x[32][21]*p[21]+x[32][22]*p[22]+x[32][23]*p[23]+x[32][24]*p[24]+x[32][25]*p[25]+x[32][26]*p[26]+x[32][27]*p[27]+x[32][28]*p[28]+x[32][29]*p[29]+x[32][30]*p[30]+x[32][31]*p[31]+x[32][32]*p[32]+x[32][33]*p[33]+x[32][34]*p[34]+x[32][35]*p[35]+x[32][36]*p[36]+x[32][37]*p[37]-1/p[32],x[33][0]*p[0]+x[33][1]*p[1]+x[33][2]*p[2]+x[33][3]*p[3]+x[33][4]*p[4]+x[33][5]*p[5]+x[33][6]*p[6]+x[33][7]*p[7]+x[33][8]*p[8]+x[33][9]*p[9]+x[33][10]*p[10]+x[33][11]*p[11]+x[33][12]*p[12]+x[33][13]*p[13]+x[33][14]*p[14]+x[33][15]*p[15]+x[33][16]*p[16]+x[33][17]*p[17]+x[33][18]*p[18]+x[33][19]*p[19]+x[33][20]*p[20]+x[33][21]*p[21]+x[33][22]*p[22]+x[33][23]*p[23]+x[33][24]*p[24]+x[33][25]*p[25]+x[33][26]*p[26]+x[33][27]*p[27]+x[33][28]*p[28]+x[33][29]*p[29]+x[33][30]*p[30]+x[33][31]*p[31]+x[33][32]*p[32]+x[33][33]*p[33]+x[33][34]*p[34]+x[33][35]*p[35]+x[33][36]*p[36]+x[33][37]*p[37]-1/p[33],x[34][0]*p[0]+x[34][1]*p[1]+x[34][2]*p[2]+x[34][3]*p[3]+x[34][4]*p[4]+x[34][5]*p[5]+x[34][6]*p[6]+x[34][7]*p[7]+x[34][8]*p[8]+x[34][9]*p[9]+x[34][10]*p[10]+x[34][11]*p[11]+x[34][12]*p[12]+x[34][13]*p[13]+x[34][14]*p[14]+x[34][15]*p[15]+x[34][16]*p[16]+x[34][17]*p[17]+x[34][18]*p[18]+x[34][19]*p[19]+x[34][20]*p[20]+x[34][21]*p[21]+x[34][22]*p[22]+x[34][23]*p[23]+x[34][24]*p[24]+x[34][25]*p[25]+x[34][26]*p[26]+x[34][27]*p[27]+x[34][28]*p[28]+x[34][29]*p[29]+x[34][30]*p[30]+x[34][31]*p[31]+x[34][32]*p[32]+x[34][33]*p[33]+x[34][34]*p[34]+x[34][35]*p[35]+x[34][36]*p[36]+x[34][37]*p[37]-1/p[34],x[35][0]*p[0]+x[35][1]*p[1]+x[35][2]*p[2]+x[35][3]*p[3]+x[35][4]*p[4]+x[35][5]*p[5]+x[35][6]*p[6]+x[35][7]*p[7]+x[35][8]*p[8]+x[35][9]*p[9]+x[35][10]*p[10]+x[35][11]*p[11]+x[35][12]*p[12]+x[35][13]*p[13]+x[35][14]*p[14]+x[35][15]*p[15]+x[35][16]*p[16]+x[35][17]*p[17]+x[35][18]*p[18]+x[35][19]*p[19]+x[35][20]*p[20]+x[35][21]*p[21]+x[35][22]*p[22]+x[35][23]*p[23]+x[35][24]*p[24]+x[35][25]*p[25]+x[35][26]*p[26]+x[35][27]*p[27]+x[35][28]*p[28]+x[35][29]*p[29]+x[35][30]*p[30]+x[35][31]*p[31]+x[35][32]*p[32]+x[35][33]*p[33]+x[35][34]*p[34]+x[35][35]*p[35]+x[35][36]*p[36]+x[35][37]*p[37]-1/p[35],x[36][0]*p[0]+x[36][1]*p[1]+x[36][2]*p[2]+x[36][3]*p[3]+x[36][4]*p[4]+x[36][5]*p[5]+x[36][6]*p[6]+x[36][7]*p[7]+x[36][8]*p[8]+x[36][9]*p[9]+x[36][10]*p[10]+x[36][11]*p[11]+x[36][12]*p[12]+x[36][13]*p[13]+x[36][14]*p[14]+x[36][15]*p[15]+x[36][16]*p[16]+x[36][17]*p[17]+x[36][18]*p[18]+x[36][19]*p[19]+x[36][20]*p[20]+x[36][21]*p[21]+x[36][22]*p[22]+x[36][23]*p[23]+x[36][24]*p[24]+x[36][25]*p[25]+x[36][26]*p[26]+x[36][27]*p[27]+x[36][28]*p[28]+x[36][29]*p[29]+x[36][30]*p[30]+x[36][31]*p[31]+x[36][32]*p[32]+x[36][33]*p[33]+x[36][34]*p[34]+x[36][35]*p[35]+x[36][36]*p[36]+x[36][37]*p[37]-1/p[36],x[37][0]*p[0]+x[37][1]*p[1]+x[37][2]*p[2]+x[37][3]*p[3]+x[37][4]*p[4]+x[37][5]*p[5]+x[37][6]*p[6]+x[37][7]*p[7]+x[37][8]*p[8]+x[37][9]*p[9]+x[37][10]*p[10]+x[37][11]*p[11]+x[37][12]*p[12]+x[37][13]*p[13]+x[37][14]*p[14]+x[37][15]*p[15]+x[37][16]*p[16]+x[37][17]*p[17]+x[37][18]*p[18]+x[37][19]*p[19]+x[37][20]*p[20]+x[37][21]*p[21]+x[37][22]*p[22]+x[37][23]*p[23]+x[37][24]*p[24]+x[37][25]*p[25]+x[37][26]*p[26]+x[37][27]*p[27]+x[37][28]*p[28]+x[37][29]*p[29]+x[37][30]*p[30]+x[37][31]*p[31]+x[37][32]*p[32]+x[37][33]*p[33]+x[37][34]*p[34]+x[37][35]*p[35]+x[37][36]*p[36]+x[37][37]*p[37]-1/p[37]
])
    return f1

if __name__ == "__main__":
    x = []
    n=38
    for i in range(n):
        x.append([])
        for j in range(1 + i, 39 + i):
            x[i].append(1)
    f1 = gen(x)
    sol_fsolve = fsolve(f1, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    print(sol_fsolve)


