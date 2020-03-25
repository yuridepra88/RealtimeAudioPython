
/* C code implementing the Diode Ring Modulator equations */

/*tecnical electical data*/
const double C   = 10e-9;  // F
const double C_p = 10e-9 ; // F
const float L   = 0.8;     // H
const int R_a = 600;       // ohm
const int R_i = 50;        // ohm
const int R_m = 80;        // ohm

double u1 = 0;
double u2 = 0;
double u3 = 0;
double u4 = 0;
double u5 = 0;
double u6 = 0;
double u7 = 0;

double i1 = 0;
double i2 = 0;

void compute_c(double* data, double* mod_signal,int samples, float T){
	
	int i=0;
	for (i = 0; i < samples; i++) {
		u1 = u1 + T/C*(i1 - (u4*u4*u4*u4*0.17)/2 + (u7*u7*u7*u7*0.17)/2 + (u5*u5*u5*u5*0.17)/2 - (u6*u6*u6*u6*0.17)/2 - (u1-mod_signal[i]*0.2)/R_m );
		u2 = u2 + T/C*(i2 + (u4*u4*u4*u4*0.17)/2 + (u7*u7*u7*u7*0.17)/2 - (u5*u5*u5*u5*0.17)/2 - (u6*u6*u6*u6*0.17)/2 - u2/R_a );
		u3 = u3 + T/C_p*((u4*u4*u4*u4*0.17) - (u7*u7*u7*u7*0.17) + (u5*u5*u5*u5*0.17) - (u6*u6*u6*u6*0.17)- (u3)/R_i );
	    

		u4 =  u1/2 - u3 - data[i]*0.2 - u2/2;
		u5 = -u1/2 - u3 - data[i]*0.2 + u2/2;
		u6 =  u1/2 + u3 + data[i]*0.2 + u2/2;
		u7 = -u1/2 + u3 + data[i]*0.2 - u2/2;

		i1 = i1+ T/L*(-u1);
		i2 = i2+ T/L*(-u2);
		data[i] = u2;
	}	    
	return;

}
