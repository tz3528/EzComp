# comp语言编译器前端

该前端用于将comp语言转换为mlir。

## 语法
comp语言由三部分构成，其大致结构如下

``` C++
declarations {
    x[0,100,101];
    t[0,100,101];
}
equations {
    diff(u(x,t),t,1)=100*diff(u(x,t),x,2);
    diff(u(x,t),x,2)=(u(x-1,t)+u(x+1,t)-2*u(x,t))/delta(x,2);
    u(x,0)=0;
    u(0,t)=10;
    u(100,t)=10;
}
options {
    mode:"time-pde";
    method:"FDM";
    precision:-6;
    timeVar:"t";
}
```

- declarations：这部分中用于声明变量，其中三个参数分别为参数下界、参数上界、区间内取的参数个数。
- equations：这部分用于声明方程。其中，在解时序方程时，默认u是待求函数，t时时间单位，用于求解。
- options：这部分用于描述选项。其中mode用于表示解方程的类型，method用于表示求解方式，precision是精度 ,等式右侧表示是10的多少次幂。
