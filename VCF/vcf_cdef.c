#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double gfdbk = 1.5;//5.175;
const double I0 = 30e-6;
const double eta = 1.836;
const double Vt = 0.026;
const double c = 0.01e-6;
const double REL_ERR = 10e-4;


double xc1 = 0;
double xc2 = 0;
double xc3 = 0;
double xc4 = 0;
double vc1 = 0;
double vc2 = 0;
double vc3 = 0;
double vc4 = 0;
double vc11 = 0;
double vc21 = 0;
double vc31 = 0;
double vc41 = 0;


double s1 = 0;
double s2 = 0;
double s3 = 0;
double s4 = 0;

double vin1= 0;
double vout = 0;
 

void compute_c(double* vin ,int samples, float T){
	double gmma = eta*Vt;

	int i=0;
	for (i = 0; i < samples; i++) {

    	float vc4 = 0;
    	float vc4Past = 1;
		int nIter = 0;

   		while( abs((vc4-vc4Past)) > (REL_ERR*abs(vc4Past)) ){    

       			vc4Past = vc4;
				vin1 = tanh((vin[i] - vout)/(2*Vt));
       
       			xc1 = (I0/2/c) * (vin1 + vc11);
      			vc1 = T/2*xc1 + s1;
       			vc11 = tanh((vc2-vc1)/(2*gmma));
       
		        xc2 = (I0/2/c) * (vc21 - vc11);
		        vc2 = T/2*xc2 + s2;
		        vc21 = tanh((vc3-vc2)/(2*gmma));

		        xc3 = (I0/2/c) * (vc31 - vc21);
		        vc3 = T/2*xc3 + s3;
		        vc31 = tanh((vc4-vc3)/(2*gmma));
		   
		        xc4 = (I0/2/c) * (-vc41 - vc31);
		        vc4 = T/2*xc4 + s4;
		        vc41 = tanh(vc4/(6*gmma));
		       
		        vout = vc4/2 + vc4*gfdbk;
       
       			nIter = nIter+1;
       			if (nIter == 100){
				nIter=0;           			
				break;
       			} 
   		}

		vin[i] = vout;
		s1 = T/2*xc1 + vc1;
		s2 = T/2*xc2 + vc2;
		s3 = T/2*xc3 + vc3;
		s4 = T/2*xc4 + vc4;
	}
	return;
}


