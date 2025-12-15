# comp语言编译器前端

该前端用于将comp语言转换为mlir。

## 语法
comp语言由三部分构成，其大致结构如下

``` C++
declarations {
    x=0;
    y=0;
}
equations {
    x=diff(x,0)+y+2;
    y=x-y-2;
}
options {
    precision:-6;
    delta:-6;
    length:1000;
}
```

- declarations：这部分中用于声明变量，其中赋的值为方程中变量的初值。若为赋初值，则默认为0。
- equations：这部分用于声明方程，其中每个变量最多被赋值一次，即一个变量最多在等式左边出现一次。若其未被赋值，则默认为常量。
- options：这部分用于描述选项。其中Precision是结果 s的精度，delta为求解步长，length为迭代次数。前两者的等式右侧均是10的多少次幂。