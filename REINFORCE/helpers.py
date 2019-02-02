import numpy as np

if __name__ == '__main__':
    a = np.random.rand(5) * 20 // 1 - 10
    print(a)

    print(discount_and_normalize_rewards(a, .975).round(3))
