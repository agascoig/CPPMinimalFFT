
# Prime Factor Algorithm (2 factors)

## n indexing

The mapping used is defined by:

$$
n = \left< N_2 n_1 + A_1 n_2 \right>_N
$$

$$
A_1 = p_1 N _1 = Q_1 N_2 + 1
$$

For the implementation, two counters are used sequentially
updated each cycle to generate $n_1'$ and $n_2'$:

$$
n_2' := \begin{cases}
n_2' + 1 & \text{if} \space n_2' \neq N_2 - 1 \\
0 & \text{otherwise}
\end{cases}
$$

The $n_1$ counter transitions on $n_2'$ wrapping to zero:

$$
n_1' := \begin{cases}
n_1' + 1 & \text{if} \space n_1' \neq N_1 - 1, \text{on} \space n_2' \rightarrow 0 \\
0 & \text{otherwise}
\end{cases}
$$

These two counters are used to generate a mapping $(n_1,n_2)$:

$$
n_1 = \left< n_1' + Q_1' n_2' \right>_{N_1} \\[0.2cm]
n_2 = n_2'
$$

where $Q_1' = \left< N_x - Q_1 \right>_{N_1}$.

An auxilliary counter $R_1$ is used to generate $n_1$ and is updated every cycle:

$$
R_1 := \begin{cases}
\left< R_1 + Q_1' \right>_{N_1} & \text{if} \space R_1 \neq N_1 - 1 \\
0 & \text{otherwise}
\end{cases}
$$

so that $n_1$ is calculated as $\left< n_1' + R_1 \right>_{N_1}$.
 
Therefore, the final mapping for input X and output Y is:

$$
Y[n_1 + N_1 n_2] = X[N_2 n_1' + n_2'] 
$$

Note that the index for X can be implemented as a single counter.

## k indexing

$$
k = \left< B_1 k_1 + N_1 k_2 \right>_N
$$

$$
B_1 = p_2 N_2 = Q_2 N_1 + 1
$$

Again, two counters are used to generate $k_1'$ and $k_2'$, as in the case of the n mapping.

These are used to generate the mapping $(k_1,k_2)$:

$$
k_1 = k_1'
$$

$$
k_2 = \left< k_2' + Q_2' k_1' \right>_{N_2}
$$

where $Q_2' = \left< N_2 - Q_2 \right>_{N_2}$.

An auxilliary counter $R_1$ is used to generate $k_2$:

$$
R_1 := \begin{cases}
0 & \text{if} \space k_1' \rightarrow 0 \\
\left< R_1 + Q_2' \right>_{N_2} & \text{otherwise}
\end{cases}
$$

so that $k_2 = \left< k_2' + R_1 \right>_{N_2}$. 

Therefore, the final mapping for input X and output Y is:

$$
Y[N_2 k_1' + k_2'] = X[k_1 + N_1 k_2]
$$

Note that the index for Y can be implemented as a single counter.

## Remarks

In this paper's notation, let $\left<A\right>_B = A \space \text{mod} \space B.$

## References

[1] A. Wang, J. Bachrach and B. Nikolié, "A generator of memory-based, runtime-reconfigurable 2N3M5K FFT engines," 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Shanghai, China, 2016, pp. 1016-1020, doi: 10.1109/ICASSP.2016.7471829

[2] Wang, Angie.  Ph.D. Dissertation, UC Berkeley.  "Agile Design of Generator-Based Signal Processing Hardware," 2018.

[3] C. -F. Hsiao, Y. Chen and C. -Y. Lee, "A Generalized Mixed-Radix Algorithm for Memory-Based FFT Processors," in IEEE Transactions on Circuits and Systems II: Express Briefs, vol. 57, no. 1, pp. 26-30, Jan. 2010, doi: 10.1109/TCSII.2009.2037262
