#include <iostream>
#include <cmath>
#include <vector>

#ifdef USE_DOUBLE
    using type_array = double;
    const double PI = 3.1415926535897932384;
#else
    using type_array = float;
    const float PI = 3.1415926535897932384f;
#endif

int main(){
    
    int n = 10000000;
    std::vector<type_array> sin_table(n);

    double sum = 0.0;

    for (int i = 0; i < n; ++i){
        sin_table[i] = std::sin(2.0 * PI * i / n);
        sum += sin_table[i];
    }

    std::cout << "Sum of sine values:: " << sum << std::endl;

    return 0;
}
