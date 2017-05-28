#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <x86intrin.h>

#define USE_AVX

inline float sigmoid(float x){
	return 1.0f/(1+std::exp(x));
}
inline float d_sigmoid(float x){
	return sigmoid(x)*(1.0f-sigmoid(x));
}

inline float threshould_func(float x){
	return x > 0.0 ? 1.0f : 0.0f;
}

class Vec8{
	float *a;
	float *a_orig;
public:
	Vec8():a_orig(new __attribute__((aligned(32))) float[8+8]){
		a = a_orig;
		while(reinterpret_cast<long>(a)%32)a++;
	}
	~Vec8(){
		delete [] a_orig;
	}
	void print(std::string str = ""){
		printf("%s(%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f)\n",str.c_str(),a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]);
	}
	float *get_ptr() const {return a;};
	void operator=(Vec8& v){
		for(int i = 0;i < 8;i++)v.a[i] = a[i];
	}
	void init(float f){
		for(int i = 0;i < 8;i++) a[i] = f;
	}
	void init(float min,float max){
		std::uniform_real_distribution<float> dist(min,max);
		std::random_device rnd;
		std::mt19937 mt(rnd());
		std::generate(a,a+8,[&dist,&mt](){return dist(mt);});
	}
	void mult(float m,Vec8& dst_vec){
		for(int i = 0;i < 8;i++){
			dst_vec.get_ptr()[i] = m * a[i];
		}
	}

	// des = this + m * src
	void add(float m,const Vec8 &src_vec,Vec8 &dst_vec){
#ifdef USE_AVX
		__m256 cons;
		for(int i = 0;i < 8;i++) cons[i] = m;
		__m256 sx = _mm256_load_ps(src_vec.get_ptr());
		__m256 sy = _mm256_load_ps(a);
		sx = _mm256_mul_ps(sx,cons);
		sx = _mm256_add_ps(sx,sy);
		_mm256_store_ps( dst_vec.get_ptr(), sx );
#else
		for(int i = 0;i < 8;i++){
			dst_vec.get_ptr()[i] = src_vec.get_ptr()[i] * m + a[i];
		}
#endif
	}
	// des = this - m * src
	void sub(float m,const Vec8 &src_vec,Vec8 &dst_vec){
		__m256 cons;
		for(int i = 0;i < 8;i++) cons[i] = m;
		__m256 sx = _mm256_load_ps(src_vec.get_ptr());
		__m256 sy = _mm256_load_ps(a);
		sx = _mm256_mul_ps(sx,cons);
		sx = _mm256_sub_ps(sx,sy);
		_mm256_store_ps( dst_vec.get_ptr(), sx );
	}

	float inner(const Vec8& src_vec) const{
#ifdef USE_AVX
		__m256 w1 = _mm256_load_ps(a);
		__m256 w2 = _mm256_load_ps(src_vec.get_ptr());
		w1 = _mm256_mul_ps(w1,w2);
		__attribute__((aligned(32)))float r[8];
		_mm256_store_ps(r,w1);
		return r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + r[6] + r[7];
#else
		float sum = 0.0f;
		for(int i = 0;i < 8;i++) sum+=a[i]*src_vec.get_ptr()[i];
		return sum;
#endif
	}
};

class Mat88{
	// __m256は連続メモリ出ないといけないのかな?
	// だとするとFortran配列ではダメっぽい?
	float *a;
	float *a_orig;
public:
	Mat88():a_orig(new __attribute__((aligned(32))) float[64+8]){
		a = a_orig;
		while(reinterpret_cast<long>(a)%32)a++;
	}
	~Mat88(){
		delete [] a_orig;
	}
	void init(float v){
		for(int i = 0;i < 64;i++)a[i]=v;
	}
	void init(float min,float max){
		std::uniform_real_distribution<float> dist(min,max);
		std::random_device rnd;
		std::mt19937 mt(/*rnd()*/ 0);
		std::generate(a,a+64,[&dist,&mt](){return dist(mt);});
	}
	void mult(const Vec8& src_vec,Vec8& dst_vec){
		__m256 u1 = {0};

		__m256 w1 = _mm256_load_ps(src_vec.get_ptr());
		__attribute__((aligned(32))) float res[8] = {0};
		for(int i = 0;i < 8;i++){
			__m256 m1 = _mm256_load_ps(a+i*8);
			m1 = _mm256_mul_ps(m1,w1);
			_mm256_store_ps(res,m1);
			dst_vec.get_ptr()[i] = [res](){float sum = 0.0f;for(int i = 0;i < 8;i++)sum+=res[i];return sum;}();
		}
	}
};

float init_vector(Vec8& vec,const Vec8& all_1){
	vec.init(-1.0f,1.0f);
	for(int i = 0;i < 8;i++){
		if( vec.get_ptr()[i] < 0.0f) vec.get_ptr()[i] = 0.0f;
		else vec.get_ptr()[i] = 1.0f;
	}
	return all_1.inner(vec) > 4 ? 0.0f : 1.0f;
}

int main(){
	Vec8 w,v,all_1;
	w.init(-0.5f,0.5f);
	all_1.init(1.0f);
	float teacher = 0.0f;
	float b = 0.0f;
	float alpha = 0.003f;
	const int CALC = 7000;
	int last_change_c = 0;
	//std::ofstream ofs("output.csv");
	//ofs<<"trial,teacher,output,error,b\n";
	auto start_time = std::chrono::system_clock::now();
	for(int c = 0;c < CALC;c++){
		teacher = init_vector( v , all_1 );
		//w.print("weight=");
		//v.print("case=");
		float output = threshould_func( w.inner( v ) + b);
		//std::cout<<"output="<<output<<std::endl;
		float error = teacher - output;
		if( error > 0.0f || error < 0.0f)
			last_change_c = c;
		//std::cout<<"error="<<error<<std::endl;
		alpha = 0.01f/std::log10(c+10.0f);
		v.mult(error*alpha,v);
		b += error*alpha;
		//v.print("delta=");
		w.add(1.0f,v,w);
		//std::cout<<"--end:"<<c<<"--"<<std::endl;
		//ofs<<c<<","<<teacher<<","<<output<<","<<error<<","<<b<<"\n";
	}
	auto stop_time = std::chrono::system_clock::now();
	w.print("weight=");
	std::cout<<"b="<<b<<std::endl;
	std::cout<<"convergence at "<<last_change_c<<std::endl;
	std::cout<<"elapsed time = "<<std::chrono::duration_cast<std::chrono::microseconds>(stop_time-start_time).count()/CALC<<" [us]"<<std::endl;
	//ofs.close();
}
