comp.problem attributes {
    mode = "time-pde", method = "FDM", precision = -6,
    function = "u(x,t)", timeVar = "t"
} {
    // 网格
    comp.dim @x domain<lower=0, upper=100, points=101, step=1.0>
    comp.dim @t domain<lower=0, upper=100, points=101, step=1.0>
    
    // 目标未知函数与维度角色
    %u = comp.field @u (spaceDims=[@x], timeDim=@t) : !comp.field<f64>
    
    // （可选）注册离散规则：diff(u,x,2) 的定义
    comp.rule "diff" (@u, @x, 2) {
        // RHS: (u(x-1,t)+u(x+1,t)-2*u(x,t))/delta(x,2)
        // 这里表达式你可以先用 comp.* 表示，后续再 canonicalize
    }
    
    // 求解流程
    comp.solve %u {
        // region 0: init
        comp.apply_init %u anchors=[#comp.anchor<dim=@t, value=0, index=0>] {
            // rhs: 0
            comp.yield 0.0 : f64
        }
    
    } boundary {
        // region 1: boundary
        comp.apply_boundary %u anchors=[#comp.anchor<dim=@x, value=0, side=Min, index=0>] {
            comp.yield 10.0 : f64
        }
        comp.apply_boundary %u anchors=[#comp.anchor<dim=@x, value=100, side=Max, index=100>] {
            comp.yield 10.0 : f64
        }
    
    } step {
        // region 2: one step (隐含会执行 Nt-1 次)
        // 这里放 iter 方程，比如：diff(u,t,1)=100*diff(u,x,2)
        comp.update %u {
            // symbolic equation region
        }
    }
}


comp.problem：问题容器（挂 options：mode/method/precision/timeVar/function）——这些 options 在源语言里就存在。

comp.dim：维度/网格信息（lower/upper/points/step）——来自 declarations 的 Domain（你 sema 已经算 step）。

comp.field：未知函数（u）+ 维度角色（timeDim/spaceDims）——来自 TargetFunctionMeta。

comp.rule：注册离散规则（例如 diff/delta 的等价展开）——因为你的源程序里本身就把 diff(u,x,2)=... 写成“定义式”。

comp.solve：求解流程 op（至少 3 个 region：init / boundary / step）

comp.apply_init、comp.apply_boundary：把 init/boundary 约束显式化，并携带 anchors（dim/value/index/side）——这些 anchors 你 sema 已经有结构化结果。

comp.update：iter 组方程（iter）。