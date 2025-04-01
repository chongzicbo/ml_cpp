#include <iostream>
#include <functional>

class Variable
{
public:
    double value;
    double grad;

    Variable(double value) : value(value), grad(0.0) {}

    Variable operator+(const Variable &other) const
    {
        Variable result(this->value + other.value);
        result.grad = this->grad + other.grad;
        return result;
    }

    Variable operator*(const Variable &other) const
    {
        Variable result(this->value * other.value);
        result.grad = this->grad * other.value + other.grad * this->value;
        return result;
    }
};

void test_autodiff()
{
    Variable x(2.0);
    Variable y(3.0);

    // Example: z = x * y + x
    Variable z = x * y + x;

    // Set gradient of z to 1 (seed for backpropagation)
    z.grad = 1.0;

    // Print results
    std::cout << "z.value: " << z.value << std::endl;
    std::cout << "z.grad: " << z.grad << std::endl;
}
