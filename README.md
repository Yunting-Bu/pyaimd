# 借助PySCF实现NVT系综下的从头算动力学
从头算动力学 ( Ab initio molecular dynamics, AIMD ) 是一种使用量子化学方法计算梯度与能量，使用分子动力学方法研究核的运动的一种动力学方法。编写 AIMD 程序的难点在于能量与梯度部分的编写，而 `PySCF` 作为一款基于 Python 的量化程序，可以为我们编写 AIMD 程序提供计算能量与梯度的接口，从而很方便的学习 AIMD 的实现方法。本文档的目的在于实现小分子在 NVT 系综下的从头算动力学计算，我们提供了简单的 Python 代码供读者参考。
## AIMD的算法
AIMD 的算法可以简单分为能量计算、梯度计算以及分子动力学三个部分。
### 能量的计算
AIMD 是通过量子化学方法得到能量，常用的方法有 Hartree-Fock 近似、密度泛函理论 (DFT) 以及 post-HF 方法，比如，在 Hartree-Fock 近似中，电子能量可以通过下面的公式得到：
$$
E_0=\dfrac{1}{2}\sum\limits_\mu\sum\limits_\nu P_{\nu\mu}(H^{core}_{\mu\nu}+F_{\mu\nu})
$$
在 MP2 方法中，则需要考虑相关能：
$$
E_{corr}=\sum\limits_{ij}\sum\limits_{ab}\dfrac{(ia|jb)[2(ia|jb)-(ib|ja)]}{\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b}
$$
更多的方法请查阅相关的量子化学教材。我们可以使用 `PySCF` 程序方便的计算出体系的能量：
```python
from pyscf import gto 
from pyscf import scf

mol = gto.Mole()
mol.atom = "H2O.xyz"
mol.basis = "sto-3g"
mol.build()  

mf = scf.RHF(mol)
Epot = mf.kernel()
```

### 梯度计算
常用的梯度计算方法有数值梯度与解析梯度两种。

数值梯度可以使用有限差分法得到，如求函数 $f(x)$ 在 $x=x_0$ 处的导数：
$$
	f^\prime(x_0)=\dfrac{f(x_0+\Delta)-f(x_0-\Delta)}{2\Delta}
$$
其中 $\Delta$ 为差分步长，理论上其值越小得到的结果约准确。但在实际计算中，$f(x_0)$ 一般不能完全精准的算出，会产生一定的数值误差，如果差分步长取得太小，可能会使结果产生浮动，影响精度。

解析梯度通过对能量求导得来，如 Hartree-Fock 方法的解析梯度为：
$$
\dfrac{\partial E}{\partial X_A}=\sum\limits_{\mu\nu}P_{\nu\mu}\dfrac{\partial H^{core}_{\mu\nu}}{\partial X_A}+\dfrac{1}{2}\sum\limits_{\mu\nu\lambda\sigma}P_{\nu\mu}P_{\lambda\sigma}\dfrac{\partial(\mu\nu||\sigma\lambda)}{\partial X_A}\nonumber\\
-\sum\limits_{\mu\nu}Q_{\nu\mu}\dfrac{\partial S_{\mu\nu}}{\partial X_A}+\dfrac{\partial V_{NN}}{\partial X_A}
$$
其中
$$
Q_{\nu\mu}=2\sum\limits^{N/2}_a\varepsilon_aC_{\mu a}C_{\nu a}
$$
解析梯度的优点是计算速度较数值梯度更快，但并不是每一种方法都有解析梯度。

在 `PySCF` 中可以很方便的计算梯度：
```Python
from pyscf import grad

g = mf.Gradients()
grad = g.kernel()
```

### 分子动力学
#### 速度初始化
在进行 MD 手续之前， 需要对速度进行初始化， 常用的初始化方法为
Maxwell-Boltzmann 分布。在给定温度 `init_temp` 下，未归一化的 MB 分布公式为：
$$
f(v)=v^2\mathrm{exp}\left( -\dfrac{mv^2}{2k_BT}\right) 
$$
为了初始化速度，还需要一个符合正态分布的随机数，其
平均值为 0 方差为 1，记为 $\mathcal{N}(0,1)$。对于每一个原子，笛卡尔坐标下速度的分量为
$$
v_{i,\alpha}=\sqrt{\dfrac{k_BT}{m_i}}\mathcal{N}(0,1),\quad\alpha\in\{x,y,z\},\quad i=1,2,\dots,N
$$
$\mathcal{N}(0,1)$ 产生通常使用 Box-Muller 变换。设$u_1, u_2$ 是 $[0,1]$ 内任意随机数，通过以下公式得到符合 $\mathcal{N}(0,1)$ 条件的随机数：
$$
z_1=\sqrt{-2\mathrm{log}(u_1)}\mathrm{cos}(2\pi u_2)\nonumber\\
	z_2=\sqrt{-2\mathrm{log}(u_1)}\mathrm{sin}(2\pi u_2)
$$
也可以使用 `numpy` 的 `np.random.normal()` 函数轻松的实现。最后，我们需要消除质心的移动，令总的动量为零来产生新的速度
$$
\mathbf{P}_{tot}=\sum\limits^{N}_{i=1}m_i\mathbf{v}_{i,old}\\
	\mathbf{v}_{i,new}=\mathbf{v}_{i,old}-\dfrac{\mathbf{P}_{tot}}{m_iN}
$$
#### 运动积分
分子动力学采用牛顿运动方程来求算分子骨架随时间的演变，牛顿运动方程为一种常微分方程，MD 中常使用 Verlet 法以及其变种进行数值求解。速度 Verlet 方法是一种较为常用且简单的方法，其运动方程为：
$$
\mathbf{v}(t+\Delta t)=\mathbf{v}(t)+\dfrac{\Delta t(\mathbf{a}(t)+\mathbf{a}(t+\Delta t))}{2}\\
	\mathbf{r}(t+\Delta t)=\mathbf{r}(t)+\Delta t\mathbf{v}(t)+\dfrac{\Delta t^2\mathbf{a}(t)}{2}
$$
其中加速度为
$$
\mathbf{a}(t)=\dfrac{\mathbf{F}(t)}{m_i}
$$
#### Berendsen 热浴
在特定温度下初始化速度后，可以假定系统以 NVE 系综进行运动积分，并且起始温度大致保持不变。然而，在真实系统中，有至少两个原因会导致温度在运动几步之后就会明显偏离初始值。首先，初始速度分布仅考虑了粒子的动能，但是一些动能会在运动开始时立即与势能贡献（例如键伸缩）交换，从而改变温度。其次，由于有限时间步长引入的数值误差会导致能量和温度漂移。为了抵消这些影响，通常希望在模拟过程中进行温度控制，使系统以 NVT 系综运行，这被称为热浴。

Berendsen 热浴具体运作原理如下：

首先计算矫正因子
$$
f=\sqrt{1+\dfrac{T_{bath}-T_{c}}{T_{c}\tau}}
$$
其中，$T_{bath}$ 为设定的热浴温度，$\tau$ 为时间常数，通常为 $20-200 fs$，$T_c$ 为使用如下公式计算的温度（非线型分子）：
$$
E_k=\dfrac{1}{2}\sum\limits_i^Nm_i\mathbf{v}_i^2
$$
$$
	T_c=\dfrac{2E_k}{3k_BN}
$$
$E_k$ 为体系动能，于是，矫正后的速度为
$$
\mathbf{v}_{i,new}=f\cdot\mathbf{v}_{i,old}
$$
要格外注意需要对单位进行替换，如将 fs 与 dalton 都转换成原子单位制。