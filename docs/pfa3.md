
# Prime Factor Algorithm (3 factors)

## n indexing

$$
n = \left< N_2 N_3 n_1 + A_1 \tilde{n}_2 \right>_N \\
\tilde{n}_2 = \left< N_3 n_2 + A_2 n_3\right>_{N_2 N_3} \\[0.4cm]
A_1 = p_1 N_1 = Q_1 N_2 N_3 + 1 \\
A_2 = p_2 N_2 = Q_2 N_3 + 1
$$

The mapping reduces to the following:

$$
n = \left< N_2 N_3 n_1 + \tilde{n}_2 \right>_N \\
\tilde{n}_2 = \left< N_3 n_2 + n_3 \right>_{N_2 N_3} \\[0.4cm]
n_1 = \left< n_1' + Q_1' \tilde{n}_2 \right>_{N_1} \\ 
n_2 = \left< n_2' + Q_2' n_3 \right>_{N_2} \\
n_3 = n_3'
$$

where $Q_x' = \left< N_x - q_x \right>_{N_x}$.

The implementation consists of three counters for $n_1',n_2',n_3'$ and
two counters $R_1$ and $R_2$.

$$
n_3' := \begin{cases}
n_3' + 1 & \text{if} \space n_3' \neq N_3 - 1 \\
0 & \text{otherwise}
\end{cases}
$$

The counter $R_2$ is defined by:

$$
R_2 := \begin{cases}
0 & \text{if} \space n_3' \rightarrow 0 \\
\left<R_2 + Q_2'\right>_{N_2} & \text{otherwise}
\end{cases}
$$

so that $n_2 = \left< n_2' + R_2\right>_{N_2}$. 

And counter $R_1$ incremented according to:

$$
R_1 := \begin{cases}
0 & \text{if} \space n_2' \rightarrow 0 \space \text{and} \space n_3' \rightarrow 0 \\
\left< R_1 + Q_1' \right>_{N_1} & \text{otherwise}
\end{cases}
$$

so that $n_1 = \left< n_1' + R_1\right>_{N_1}$.

Therefore, the final mapping for FFT input X and output Y is:

$$
Y[n_1 + N_1 n_2 + N_1 N_2 n_3] = X[N_2 N_3 n_1' + N_3 n_2' + n_3']
$$

The index for X can be implemented as just a single counter.

## k indexing

$$
k = \left< B_1 k_1 + N_1 \tilde{n}_2 \right>_N \\
\tilde{k}_2 = \left< B_2 k_2 + N_2 k_3 \right>_{N_2 N_3}\\[0.4cm]
B_1 = p_4 N_2 N_3 = Q_4 N_1 + 1 \\
B_2 = p_3 N_3 = Q_3 N_1 N_2 + 1 \\
(p_4 = p_2 p_3)
$$

The mapping reduces to the following:

$$
k = \left< N_2 N_3 k_1 + \tilde{k}_2 \right>_N\\
\tilde{k}_2 = \left< N_3 k_2 + k_3 \right>_{N_2 N_3}\\[0.4cm]
k_1 = k_1' \\
k_2 = \left< k_2' + \left< Q_3' k_1' \right>_{N_2} \right>_{N_2} \\
k_3 = \left< k_3' + \left< Q_4' (k_1' + N_1 k_2') \right>_{N_3} \right>_{N_3}
$$

where $Q_3' = \left< N_3 - Q_3 \right>_{N_3}$ and $Q_4' = \left< N_2 - Q_4\right>_{N_2}$.

The implementation consists three counters for $(k_1',k_2',k_3')$ 
similar to the n mapping, and $R_1$ and $R_2$.

$$
R_1 := \begin{cases}
0 & \text{if} \space n_2' \rightarrow 0 \space \text{and} \space n_3' \rightarrow 0 \\
\left< R_1 + Q_1' \right>_{N_1} & \text{otherwise}
\end{cases}
$$

$$
R_2 := \begin{cases}
0 & \text{if} \space n_3' \rightarrow 0 \\
\left< R_2 + Q_2' \right>_{N_2} & \text{otherwise}
\end{cases}
$$

Then $k_1 = \left< k_1' + R_1 \right>$ and $k_2 = \left< n_2' + R_1 \right>$

The final mapping for FFT input X and output Y is:

$$
Y[N_2 N_3 k_1' + N_3 k_2' + k_3'] = X[k_1 + N_1 k_2 + N_1 N_2 k_3]
$$

Note that the index for Y can be implemented as just a single counter.

## Remarks

In this paper's notation, let $\left<A\right>_B = A \space \text{mod} \space B.$.

## References

[1] A. Wang, J. Bachrach and B. Nikolié, "A generator of memory-based, runtime-reconfigurable 2N3M5K FFT engines," 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Shanghai, China, 2016, pp. 1016-1020, doi: 10.1109/ICASSP.2016.7471829

[2] Wang, Angie.  Ph.D. Dissertation, UC Berkeley.  "Agile Design of Generator-Based Signal Processing Hardware," 2018.

[3] C. -F. Hsiao, Y. Chen and C. -Y. Lee, "A Generalized Mixed-Radix Algorithm for Memory-Based FFT Processors," in IEEE Transactions on Circuits and Systems II: Express Briefs, vol. 57, no. 1, pp. 26-30, Jan. 2010, doi: 10.1109/TCSII.2009.2037262
