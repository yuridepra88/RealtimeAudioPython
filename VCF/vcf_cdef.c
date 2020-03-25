#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double gfdbk = 1.5;//5.175;
const float I0 = 30e-6;

const float eta = 1.836;
const float Vt = 0.026;


const float c = 0.01e-6;
const float REL_ERR = 10e-4;


float xc1 = 0;
float xc2 = 0;
float xc3 = 0;
float xc4 = 0;
float vc1 = 0;
float vc2 = 0;
float vc3 = 0;
float vc4 = 0;
float vc11 = 0;
float vc21 = 0;
float vc31 = 0;
float vc41 = 0;


float s1 = 0;
float s2 = 0;
float s3 = 0;
float s4 = 0;

float vin1Temp = 0;
float voutTemp = 0;
 
float vc1Temp = 0;
float vc2Temp = 0;
float vc3Temp = 0;
float vc4Temp = 0;

float vc11Temp = 0;
float vc21Temp = 0;
float vc31Temp = 0;
float vc41Temp = 0;


float xc1Temp = 0;
float xc2Temp = 0;
float xc3Temp = 0;
float xc4Temp = 0;



void compute_c(double* vin ,int samples, float T){

	float gmma = eta*Vt;
	
	//gfdbk = gfdbk*1.5;
	int i=0;
	for (i = 0; i < samples; i++) {


    		float vc4Temp = 0;
    		float vc4Past = 1;
		int nIter = 0;
   
 
   		while( abs(vc4Temp-vc4Past) > REL_ERR*abs(vc4Past) ){    

       			vc4Past = vc4Temp ;
       
       			vin1Temp = tanh((vin[i] - voutTemp)/(2*Vt));
       
       			xc1Temp = (I0/2/c) * (vin1Temp + vc11Temp);
      			vc1Temp = T/2*xc1Temp + s1;
       			vc11Temp = tanh((vc2Temp-vc1Temp)/(2*gmma));
       
		        xc2Temp = (I0/2/c) * (vc21Temp - vc11Temp);
		        vc2Temp = T/2*xc2Temp + s2;
		        vc21Temp = tanh((vc3Temp-vc2Temp)/(2*gmma));

		        xc3Temp = (I0/2/c) * (vc31Temp - vc21Temp);
		        vc3Temp = T/2*xc3Temp + s3;
		        vc31Temp = tanh((vc4Temp-vc3Temp)/(2*gmma));
		   
		        xc4Temp = (I0/2/c) * (-vc41Temp - vc31Temp);
		        vc4Temp = T/2*xc4Temp + s4;
		        vc41Temp = tanh(vc4Temp/(6*gmma));
		       
		        voutTemp = vc4Temp/2 + vc4Temp*gfdbk;
       
       			nIter = nIter+1;
       			if (nIter == 100){
				nIter=0;           			
				break;
       			}
       
   		}

  
		   xc1 = xc1Temp;
		   xc2 = xc2Temp;
		   xc3 = xc3Temp;
		   xc4 = xc4Temp;
		   vc1 = vc1Temp;
		   vc2 = vc2Temp;
		   vc3 = vc3Temp;
		   vc4 = vc4Temp;
		   vc11 = vc11Temp;
		   vc21 = vc21Temp;
		   vc31 = vc31Temp;
		   vc41 = vc41Temp;   
		   vin[i] = voutTemp;


   
		   s1 = T/2*xc1 + vc1;
		   s2 = T/2*xc2 + vc2;
		   s3 = T/2*xc3 + vc3;
		   s4 = T/2*xc4 + vc4;

	}

	return;
}


