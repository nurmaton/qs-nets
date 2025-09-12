import numpy as np
from scipy.optimize import root
import time
import sys
import mpmath as mp

start_time = time.time()

def cos(x):
    return np.cos(np.deg2rad(x, dtype = 'float64'))

def sin(x):
    return np.sin(np.deg2rad(x, dtype = 'float64'))

def tan(x):
    return np.tan(np.deg2rad(x, dtype = 'float64'))

def is_unique(solution, solutions_list, atol=1e-6):
    for sol in solutions_list:
        if np.allclose(solution, sol, atol=atol):
            return False
    return True

def conditional_round(number, tolerance=1e-4):
    rounded = round(number)
    if abs(number - rounded) <= tolerance:
        return rounded
    else:
        return number
                    
def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def progress_bar(iteration, total, start_time, bar_length=50):
    percent = (iteration / total) * 100
    elapsed_time = time.time() - start_time
    formatted_time = format_time(elapsed_time)
    bar = '█' * int(bar_length * iteration // total) + '-' * (bar_length - int(bar_length * iteration // total))
    sys.stdout.write(f'\r|{bar}| {percent:.2f}% Complete | Time Elapsed: {formatted_time}\n')
    sys.stdout.flush()

def coefficients():
    A = np.zeros((4, 4), dtype=np.float64)
    B = np.zeros((4, 4), dtype=np.float64)
    C = np.zeros(4, dtype=np.float64)
        
    for i in range(4):
        sinThetasProduct = sin(Thetas[i]) * sin(Thetas[i - 1])
        cosGivenDeltas = cos(givenDeltas[i])
        C[i] = 4 * sinThetasProduct**2
        for j in range(4):
            A[i, j] = 4 * cos(Thetas[i] / 2 + 180 * (i + 1) * (j + 1) * j / 4 + 180 * (j + 1) / 2)**2 * cos(Thetas[i - 1] / 2 + 180 * i * (j + 1) * j / 4 + 180 * (j + 1) / 2)**2 
        
        if conditional_round(cosGivenDeltas) == 0:
            B[i, 0] = 1
            B[i, 1] = -1
            B[i, 2] = -1
            B[i, 3] = 1
        else:
            B[i, 0] = A[i, 0] * cosGivenDeltas - sinThetasProduct
            B[i, 1] = A[i, 1] * cosGivenDeltas + sinThetasProduct
            B[i, 2] = A[i, 2] * cosGivenDeltas + sinThetasProduct
            B[i, 3] = A[i, 3] * cosGivenDeltas - sinThetasProduct
    
    return A, B, C

def condition_N0(vars):
    alpha, beta, gamma, delta, sigma = vars
    
    parameter = True
    if 0 < alpha < 180 and 0 < beta < 180 and 0 < gamma < 180 and 0 < delta < 180:
        parameter = True
    else:
        parameter = False
        
    for e1 in [-1, 1]:
        for e2 in [-1, 1]:
            for e3 in [-1, 1]:
                if np.isclose((alpha + e1 * beta + e2 * gamma + e3 * delta) % (2 * np.pi), 0, atol =1e-6):
                    parameter = False
                    
    return parameter

def condition_N5(Alphas, Betas, Gammas, Deltas, Xs, Ys, Zs, u):
    def inverse_jacobi_dn_complex(x, m):
        mp.mp.prec = 100  # High precision
        # Convert inputs to mpmath types
        x = mp.mpc(x)
        m_real = mp.mpf(m)  # Ensure m is real

        # Compute periods
        K = mp.ellipk(m_real)
        Kprime = mp.ellipk(1 - m_real)

        # Calculate phi = arcsin(sqrt((1 - x²)/m))
        x_sq = x**2
        one = mp.mpf(1)
        sqrt_arg = (one - x_sq) / m_real
        sqrt_val = mp.sqrt(sqrt_arg)
        phi = mp.asin(sqrt_val)

        # Compute u = F(phi, m_real)
        inverse_dn = mp.ellipf(phi, m_real)

        real_part = mp.fmod(inverse_dn.real, 2 * K)
        imag_part = mp.fmod(inverse_dn.imag, 4 * Kprime)
        if mp.isnan(real_part) or mp.isnan(imag_part):
            inverse_dn_normalized = None
        else:
            inverse_dn_normalized = mp.mpc(real_part, imag_part)

        return inverse_dn_normalized

    def calculate_elliptic_params(M):
        if M < 1:
            m = 1 - M
        else:
            m = (M - 1) / M
            
        K = mp.ellipk(m)
        Kprime = mp.ellipk(1 - m)

        return m, K, Kprime

    def calculate_t(f, M, m):
        if M > 1:
            dn_value = 1 / mp.sqrt(f)
        else:
            dn_value = mp.sqrt(f)
            
        t = inverse_jacobi_dn_complex(dn_value, m)

        return t

    def handle_complex_values(t, K, Kprime, re_condition):

        real_ratio = np.float64(mp.re(t)) / np.float64(K)
        imag_ratio = np.float64(mp.im(t)) / np.float64(Kprime)

        if np.isclose(conditional_round(real_ratio) % 2, re_condition % 2, atol=1e-4):
            adjusted_real = re_condition * np.float64(K)
        else:
            adjusted_real = real_ratio * np.float64(K)
            
        def adjust_imaginary(imag_ratio):
            if imag_ratio < 0:
                imag_ratio += 4
                
            if 2 <= imag_ratio < 4:
                imag_ratio = 4 - imag_ratio

            return imag_ratio

        adjusted_imag = adjust_imaginary(imag_ratio) * np.float64(Kprime)
        t_adjusted = mp.mpc(adjusted_real, adjusted_imag)

        return t_adjusted


    def t_values(M, alpha, beta, gamma, delta, r, s, f):

        sigma = (alpha + beta + gamma + delta) / 2

        m, K, Kprime = calculate_elliptic_params(M)

        re_condition = None 
        if sigma < 180:
            if r > 1 and s > 1:
                re_condition = 0
            elif (r > 1 and s < 1) or (r < 1 and s > 1):
                re_condition = 1
            elif r < 1 and s < 1:
                re_condition = 2
        elif sigma > 180:
            if r > 1 and s > 1:
                re_condition = 2
            elif (r > 1 and s < 1) or (r < 1 and s > 1):
                re_condition = 3
            elif r < 1 and s < 1:
                re_condition = 0

        if re_condition is None:
            return None
        
        t = calculate_t(f, M, m)
        
        if t is not None:
            t_adjusted = handle_complex_values(t, K, Kprime, re_condition)
        else:
            t_adjusted = None

        return t_adjusted


    M = 1 - u
    Rs = [1 + 1 / x for x in Xs]
    Ss = [1 + 1 / y for y in Ys]
    Fs = [1 + 1 / z for z in Zs]
    m, K, Kprime = calculate_elliptic_params(M)
    t_results = []

    for i in range(4):
        alphai = Alphas[i]
        betai = Betas[i]
        gammai = Gammas[i]
        deltai = Deltas[i]

        ri = Rs[i]  
        si = Ss[i]

        fi = Fs[i]

        t = t_values(M, alphai, betai, gammai, deltai, ri, si, fi)
    
        t_results.append(t)
    
    if len(t_results) != 4:
        return False, None, None, None, None
    
    K = np.float64(K)
    Kprime = np.float64(Kprime)
    
    period_real = 4 * K      
    period_imag = 2 * Kprime      

    ts_info = ""
    for idx, t in enumerate(t_results, start=1):
        if t is not None:
            real_coeff = np.float64(mp.re(t)) / np.float64(K)
            imag_coeff = np.float64(mp.im(t)) / np.float64(Kprime)
            ts_info += f"t{idx} = {t} = {conditional_round(real_coeff)} * K + {conditional_round(imag_coeff)} * iK'\n"
        else:
            continue
    
    isCombinationExists = False
    es = []
    tilde_epsilon = None
    if None not in t_results:
        real_coeff1 = conditional_round(np.float64(mp.re(t_results[0])) / np.float64(K))
        imag_coeff1 = conditional_round(np.float64(mp.im(t_results[0])) / np.float64(Kprime))
        real_coeff2 = conditional_round(np.float64(mp.re(t_results[1])) / np.float64(K))
        imag_coeff2 = conditional_round(np.float64(mp.im(t_results[1])) / np.float64(Kprime))
        real_coeff3 = conditional_round(np.float64(mp.re(t_results[2])) / np.float64(K))
        imag_coeff3 = conditional_round(np.float64(mp.im(t_results[2])) / np.float64(Kprime))
        real_coeff4 = conditional_round(np.float64(mp.re(t_results[3])) / np.float64(K))
        imag_coeff4 = conditional_round(np.float64(mp.im(t_results[3])) / np.float64(Kprime))
        for i_sign in [-1, 1]:
            for j_sign in [-1, 1]:
                for k_sign in [-1, 1]:
                    real_sum = real_coeff1 + i_sign * real_coeff2 + j_sign * real_coeff3 + k_sign * real_coeff4
                    imag_sum = imag_coeff1 + i_sign * imag_coeff2 + j_sign * imag_coeff3 + k_sign * imag_coeff4
                    
                    # real_sum = float(
                    #     mp.re(t_results[0]) +
                    #     i_sign * mp.re(t_results[1]) +
                    #     j_sign * mp.re(t_results[2]) +
                    #     k_sign * mp.re(t_results[3])
                    # )
                    # imag_sum = float(
                    #     mp.im(t_results[0]) +
                    #     i_sign * mp.im(t_results[1]) +
                    #     j_sign * mp.im(t_results[2]) +
                    #     k_sign * mp.im(t_results[3])
                    # )
                    
                    if M < 1:
                        real_remainder = conditional_round(real_sum) % 4
                        imag_remainder = conditional_round(imag_sum) % 2
                        
                        if np.isclose(real_remainder, 0, atol=1e-5) and np.isclose(imag_remainder, 0, atol=1e-5):
                            isCombinationExists = True 
                            es.append(i_sign)
                            es.append(j_sign)
                            es.append(k_sign)
                            z2 = int(conditional_round(imag_sum) // 2)
                            tilde_epsilon = 1 if z2 % 2 == 0 else -1

                    else:
                        imag_remainder = conditional_round(imag_sum) % 2
                        if np.isclose(imag_remainder, 0, atol=1e-5):
                            z2 = int(conditional_round(imag_sum) // 2)
                            real_adjusted = real_sum - 2 * z2
                            real_remainder = conditional_round(real_adjusted) % 4

                            if np.isclose(real_remainder, 0, atol=1e-5):
                                isCombinationExists = True 
                                es.append(i_sign)
                                es.append(j_sign)
                                es.append(k_sign)
                                tilde_epsilon = 1 if z2 % 2 == 0 else -1
                            
    
        return isCombinationExists, t_results, ts_info, es, tilde_epsilon
    else:
        return False, None, None, None, None



def equations_squared(vars):
    x1, x3, y1, y2, z1, z2, z3, z4, u = vars
    if givenU is not None:
        u = givenU
    
    X = [x1, x1, x3, x3]
    Y = [y1, y2, y2, y1]
    Z = [z1, z2, z3, z4]
    Equations = []
    
    for i in range(4):
        eq1 = B[i, 0] + B[i, 1] * Y[i] * Z[i] * u + B[i, 2] * X[i] * Z[i] * u + B[i, 3] * X[i] * Y[i] * u
        Equations.append(eq1)
        eq2 = (A[i, 0] + A[i, 1] * Y[i] * Z[i] * u + A[i, 2] * X[i] * Z[i] * u + A[i, 3] * X[i] * Y[i] * u)**2 - X[i] * Y[i] * u * (1 + Z[i]) * (1 + u * Z[i]) * C[i]
        Equations.append(eq2)
    
    eq9 = 0
    Equations.append(eq9)
    
    return np.array(Equations, dtype=np.float64)


def generate_random_variables(iteration):
    rng = np.random.default_rng(seed=iteration)
    
    def generate_r_s(M, dihedral_angle):
        if dihedral_angle != 0 and dihedral_angle != 360:
            if dihedral_angle != 180:
                par1 = 1 / (cos(dihedral_angle / 2))**2
                par2 = 1 / (sin(dihedral_angle / 2))**2
                par3 = (1 - M / par1) * par2
                if M > 1:
                    if M < par1:
                        if M >= par2:
                            if par2 < par1:
                                parameter = rng.uniform(0, par3)
                            else:
                                parameter = None
                        else:
                            if rng.random() <= 0.5:
                                parameter = rng.uniform(0, par3)
                            else:
                                parameter = rng.uniform(M, par2)
                    else:
                        if M > par2:
                            parameter = None
                        else:
                            if par1 <= par2:
                                parameter = rng.uniform(M, par2)
                            else:
                                parameter = None
                else:
                    if M < par1:
                        if rng.random() <= 0.5:
                            parameter = rng.uniform(0, M)
                        else:
                            parameter = rng.uniform(par3, par2)
                    else:
                        parameter = None
            else:
                parameter = rng.uniform(0, np.min([1, M]))
        else:
            if M > 1:
                parameter = rng.exponential(scale=2) + M
            else:
                parameter = rng.uniform(0, M)
        
        if parameter is None:
            parameter = 1
            
        return parameter
    
    
    # def generate_f(r, s, M):
    #     if r is None or s is None or M is None:
    #         f = None
    #     else:
    #         m_left = np.min([1, M])
    #         m_right = np.max([1, M])
    #         f_denominator = M - s + r * (s - 1)
    #         if f_denominator != 0:
    #             part1 = np.sqrt(r * s * (1 - M)**2)
    #             part2 = np.sqrt((r - 1) * (r - M) * (s - 1) * (s - M))
    #             f_left = ((part1 - part2) / f_denominator)**2
    #             f_right = ((part1 + part2) / f_denominator)**2
                
    #             if f_left < m_left:
    #                 if f_right < m_left:
    #                     f = rng.uniform(f_left, f_right)
    #                 elif f_right < m_right:
    #                     f = rng.uniform(f_left, m_left)
    #                 else:
    #                     if rng.random() <= 0.5:
    #                         f = rng.uniform(f_left, m_left)
    #                     else:
    #                         f = rng.uniform(m_right, f_right)
    #             elif f_left < m_right:
    #                 if f_right < m_right:
    #                     f = None
    #                 else:
    #                     f = rng.uniform(m_right, f_right)
    #             else:
    #                 f = rng.uniform(f_left, f_right)
                
    #         else:
    #             f = rng.exponential(scale=2) + (r + s)**2 / (4 * r * s)
    #     if f is None:
    #         f = 1
            
    #     return f  
    
    def generate_f(M):
        if M is None:
            f = 1
        m_left = np.min([1, M])
        m_right = np.max([1, M])
        if rng.random() <= 0.5:
            f = rng.uniform(0, m_left)
        else:
            f = rng.exponential(scale=2) + m_right
            
        return f
    
    # def generate_f(M, r, s):
    #     if M is None:
    #         f = 1
        
    #     if M < 1:
    #         if (r - 1) * (s - 1) > 0:
    #             f = rng.exponential(scale=2) + 1
    #         elif (r - 1) * (s - 1) < 0:
    #             f = rng.uniform(0, M)
    #         else:
    #             f = 1
    #     elif M > 1:
    #         if (r - 1) * (s - 1) > 0:
    #             f = rng.uniform(0, 1)
    #         elif (r - 1) * (s - 1) < 0:
    #             f = rng.exponential(scale=2) + M
    #         else:   
    #             f = 1
    #     else:   
    #         f = 1
            
    #     return f
        
    
    while True:
        if givenU is None:
            if rng.random() <= 0.5:
                M = rng.uniform(0, 1)
            else:
                M = 1 + rng.exponential(scale=2)
        else:
            M = 1 - givenU
            
        r1 = generate_r_s(M, phi)
        s2 = generate_r_s(M, psi2)
        r3 = generate_r_s(M, theta)
        s1 = generate_r_s(M, psi1)
        
        # f1 = generate_f(r1, s1, M)
        # f2 = generate_f(r1, s2, M)
        # f3 = generate_f(r3, s2, M)
        # f4 = generate_f(r3, s1, M)
        
        # f1 = generate_f(M, r1, s1)
        # f2 = generate_f(M, r1, s2)
        # f3 = generate_f(M, r3, s2)        
        # f4 = generate_f(M, r3, s1)
        
        f1 = generate_f(M)
        f2 = generate_f(M)
        f3 = generate_f(M)        
        f4 = generate_f(M)
         
        condition1 = r1 != 1 and s2 != 1 and r3 != 1 and s1 != 1 and M != 1
        
        condition2 = f1 != 1 and f2 != 1 and f3 != 1 and f4 != 1
        condition3 = (r1 - 1) * (s1 - 1) * (f1 - 1) * (1 - M) > 0 and (r1 - 1) * (s2 - 1) * (f2 - 1) * (1 - M) > 0 and (r3 - 1) * (s2 - 1) * (f3 - 1) * (1 - M) > 0 and (r3 - 1) * (s1 - 1) * (f4 - 1) * (1 - M) > 0
        
        if condition1 and condition2 and condition3:
            break
        
    # while True:
    #     # f1 = generate_f(r1, s1, M)
    #     # f2 = generate_f(r1, s2, M)
    #     # f3 = generate_f(r3, s2, M)
    #     # f4 = generate_f(r3, s1, M)
        
    #     f1 = generate_f(M)
    #     f2 = generate_f(M)
    #     f3 = generate_f(M)        
    #     f4 = generate_f(M)
        
    #     condition2 = f1 != 1 and f2 != 1 and f3 != 1 and f4 != 1
    #     condition3 = (r1 - 1) * (s1 - 1) * (f1 - 1) * (1 - M) > 0 and (r1 - 1) * (s2 - 1) * (f2 - 1) * (1 - M) > 0 and (r3 - 1) * (s2 - 1) * (f3 - 1) * (1 - M) > 0 and (r3 - 1) * (s1 - 1) * (f4 - 1) * (1 - M) > 0
        
    #     if condition2 and condition3:
    #         break
    
    u = 1 - M
    x1 = 1 / (r1 - 1)
    x3 = 1 / (r3 - 1)
    
    y1 = 1 / (s1 - 1)
    y2 = 1 / (s2 - 1)
    
    z1 = 1 / (f1 - 1)
    z2 = 1 / (f2 - 1)
    z3 = 1 / (f3 - 1)
    z4 = 1 / (f4 - 1)
    
    return x1, x3, y1, y2, z1, z2, z3, z4, u


def get_angles(x, y, z, u, varepsilon):
    cosAlpha = varepsilon * (1 - y * z * u + x * z * u - x * y * u) / (2 * np.sqrt(x * z * u * (1 + y) * (1 + u * y)))
    cosGamma = varepsilon * (1 + y * z * u - x * z * u - x * y * u) / (2 * np.sqrt(y * z * u * (1 + x) * (1 + u * x)))
    cosDelta = varepsilon * (1 - y * z * u - x * z * u + x * y * u) / (2 * np.sqrt(x * y * u * (1 + z) * (1 + u * z)))
    cosSigma = (1 - u * (x * y + x * z + y * z + 2 * x * y * z)) / (2 * np.sqrt(x * y * z * u**2 * (1 + x) * (1 + y) * (1 + z)))
    cosBeta = varepsilon * (u * (1 + x) * (1 + y) * (1 + z) + (1 + u * x) * (1 + u * y) * (1 + u * z) - u * x * y * z * (u - 1)**2) / (2 * np.sqrt(u * (1 + x) * (1 + y) * (1 + z) * (1 + u * x) * (1 + u * y) * (1 + u * z))) 
    
    alpha = np.degrees(np.arccos(cosAlpha))
    beta = np.degrees(np.arccos(cosBeta))
    gamma = np.degrees(np.arccos(cosGamma))
    delta = np.degrees(np.arccos(cosDelta))
    sigma = np.degrees(np.arccos(cosSigma))
    
    sigmaError = np.abs((alpha + beta + gamma + delta) / 2 - sigma)
    
    append = False
    if np.isclose(sigmaError, 0, atol=1e-6) and condition_N0([alpha, beta, gamma, delta, sigma]):
        append = True

    return append, alpha, beta, gamma, delta, (alpha + beta + gamma + delta) / 2
        


def get_solutions_equations_squared(iterations = 200):
    unique_solutions = []
    
    for i in range(iterations):
        progress_bar(i + 1, iterations, start_time)  
        time.sleep(0.05)
        
        initial_guess = generate_random_variables(iteration=i)
        
        solution = root(
            equations_squared,
            initial_guess,
            method = 'lm'
            )
        
        
        x1, x3, y1, y2, z1, z2, z3, z4, u = solution.x
        if u < 1:
            if np.all(np.isclose(equations_squared(solution.x), 0, atol=1e-6)):
                if is_unique(solution.x, unique_solutions, atol=1e-3):
                    X = [x1, x1, x3, x3]
                    Y = [y1, y2, y2, y1]
                    Z = [z1, z2, z3, z4]
                    good = []
                    for i in range(4):
                        if X[i] * Z[i] * u * (1 + Y[i]) * (1 + u * Y[i]) > 0 and Y[i] * Z[i] * u * (1 + X[i]) * (1 + u * X[i]) > 0 and X[i] * Y[i] * u * (1 + Z[i]) * (1 + u * Z[i]) > 0 and X[i] * Y[i] * Z[i] * (1 + X[i]) * (1 + Y[i]) * (1 + Z[i]) > 0 and -1 < (1 - Y[i] * Z[i] * u + X[i] * Z[i] * u - X[i] * Y[i] * u) / np.sqrt(4 * X[i] * Z[i] * u * (1 + Y[i]) * (1 + u * Y[i])) < 1 and -1 < ((2 + X[i] + X[i] * u) * (1 + Y[i] * u + Z[i] * u + Y[i] * Z[i] * u) + (1 + X[i]) * (1 + Y[i] * Z[i] * u) * (u - 1)) / np.sqrt(4 * (1 + X[i]) * (1 + Y[i]) * (1 + Z[i]) * u * (1 + u * X[i]) * (1 + u * Y[i]) * (1 + u * Z[i])) < 1 and -1 <  (1 + Y[i] * Z[i] * u - X[i] * Z[i] * u - X[i] * Y[i] * u) / np.sqrt(4 * Y[i] * Z[i] * u * (1 + X[i]) * (1 + u * X[i])) < 1 and -1 < (1 - Y[i] * Z[i] * u - X[i] * Z[i] * u + X[i] * Y[i] * u) / np.sqrt(4 * X[i] * Y[i] * u * (1 + Z[i]) * (1 + u * Z[i])) < 1 and -1 <  (1 - u * (X[i] * Y[i] + X[i] * Z[i] + Y[i] * Z[i] + 2 * X[i] * Y[i] * Z[i])) / np.sqrt(4 * u**2 * X[i] * Y[i] * Z[i] * (1 + X[i]) * (1 + Y[i]) * (1 + Z[i])) < 1:
                            good.append(True)
                        else:
                            good.append(False)
                            break
                    if all(good):
                        unique_solutions.append(solution.x)
                        
    return unique_solutions


def get_clean_solutions(dirty_solutions):
    clean_solutions = []
    option = 0
    appendCounder = 0 
    satisfiedCounter = 0
    notSatisfiedCounter = 0
    for el in dirty_solutions:
        x1, x3, y1, y2, z1, z2, z3, z4, u = el
        
        X = [x1, x1, x3, x3]
        Y = [y1, y2, y2, y1]
        Z = [z1, z2, z3, z4]
        M = 1 - u
        
        epsilons = []
        alphas = []
        betas = []
        gammas = []
        deltas = []
        sigmas = []
        append = True
        for i in range(4):
            if ((1 - Y[i] * Z[i] * u - X[i] * Z[i] * u + X[i] * Y[i] * u) * cos(givenDeltas[i]) > 0 and (A[i, 0] + A[i, 1] * Y[i] * Z[i] * u + A[i, 2] * X[i] * Z[i] * u + A[i, 3] * X[i] * Y[i] * u) * sin(Thetas[i]) * sin(Thetas[i - 1]) >= 0) or ((1 - Y[i] * Z[i] * u - X[i] * Z[i] * u + X[i] * Y[i] * u) * cos(givenDeltas[i]) >= 0 and (A[i, 0] + A[i, 1] * Y[i] * Z[i] * u + A[i, 2] * X[i] * Z[i] * u + A[i, 3] * X[i] * Y[i] * u) * sin(Thetas[i]) * sin(Thetas[i - 1]) > 0):                        
                epsilons.append(1)
                append, alphai, betai, gammai,deltai, sigmai = get_angles(X[i], Y[i], Z[i], u, 1)
                if append:
                    alphas.append(alphai)
                    betas.append(betai)
                    gammas.append(gammai)
                    deltas.append(deltai)
                    sigmas.append(sigmai)
                else:
                    break
                
            elif ((1 - Y[i] * Z[i] * u - X[i] * Z[i] * u + X[i] * Y[i] * u) * cos(givenDeltas[i]) < 0 and (A[i, 0] + A[i, 1] * Y[i] * Z[i] * u + A[i, 2] * X[i] * Z[i] * u + A[i, 3] * X[i] * Y[i] * u) * sin(Thetas[i]) * sin(Thetas[i - 1]) <= 0) or ((1 - Y[i] * Z[i] * u - X[i] * Z[i] * u + X[i] * Y[i] * u) * cos(givenDeltas[i]) <= 0 and (A[i, 0] + A[i, 1] * Y[i] * Z[i] * u + A[i, 2] * X[i] * Z[i] * u + A[i, 3] * X[i] * Y[i] * u) * sin(Thetas[i]) * sin(Thetas[i - 1]) < 0):
                epsilons.append(-1)
                append, alphai, betai, gammai, deltai, sigmai = get_angles(X[i], Y[i], Z[i], u, -1)
                if append:
                    alphas.append(alphai)
                    betas.append(betai)
                    gammas.append(gammai)
                    deltas.append(deltai)
                    sigmas.append(sigmai)
                else:
                    break
                    
            else:
                append = False
                break                
    
        if append:
            appendCounder += 1
            option += 1
            satisfied, t_results, ts_info, Es, tilde_epsilon = condition_N5(alphas, betas, gammas, deltas, X, Y, Z, u)
    
            if satisfied:
                satisfiedCounter += 1
                info = f"SOLUTION {option}\nConfirmed: {satisfied}\nAlphas = {alphas}\nBetas = {betas}\nGammas = {gammas}\nDeltas = {deltas}\nSigmas = {sigmas}\nDihedralAngles = {Thetas}\n\nEpsilons = {epsilons}\nEs = {Es}\nTildeEpsilon[(-1)^n, where n = Im(t1 + e1t2 + e2t3 + e3t4) mod 2K']= {tilde_epsilon}\n\nr1 = {(x1 + 1) / x1}; r2 = {(x1 + 1) / x1}; r3 = {(x3 + 1) / x3}; r4 = {(x3 + 1) / x3}\ns1 = {(y1 + 1) / y1}; s2 = {(y2 + 1) / y2}; s3 = {(y2 + 1) / y2}; s4 = {(y1 + 1) / y1}\nf1 = {(z1 + 1) / z1}; f2 = {(z2 + 1) / z2}; f3 = {(z3 + 1) / z3}; f4 = {(z4 + 1) / z4}\nM = {M}\n\nx1 = {x1}; x2 = {x1}; x3 = {x3}; x4 = {x3}\ny1 = {y1}; y2 = {y2}; y3 = {y2}; y4 = {y1}\nz1 = {z1}; z2 = {z2}; z3 = {z3}; z4 = {z4}\nu = {u}\n\n{ts_info}\n\n\n"
                to_append = [info, alphas, betas, gammas, deltas, sigmas, Thetas, epsilons, Es, tilde_epsilon, t_results, el]
                clean_solutions.append(to_append)
            else:
                notSatisfiedCounter += 1
                
    filtered_solutions = [sol for sol in clean_solutions if not any(any(x < 20 or x > 160 for x in lst) for lst in sol[1:5])]
    garbage_solutions = [sol for sol in clean_solutions if any(any(x < 20 or x > 160 for x in lst) for lst in sol[1:5])]
    
    print(f"appendCounder = {appendCounder}, satisfiedCounter = {satisfiedCounter}, notSatisfiedCounter = {notSatisfiedCounter}")
   
    return filtered_solutions, garbage_solutions 


def main():     
    global givenDeltas
    givenDeltas = [100, 95, 110, 55]
    
    global Thetas
    # Thetas = [120, 140, 110, 130]
    # Thetas = [150, 140, 110, 130]
    # Thetas = [130, 140, 125, 135]
    # Thetas = [90, 95, 100, 105]
    Thetas = [130, 140, 120, 135]
    
    global givenU
    # givenU = 0.25
    givenU = None

    global phi, psi2, theta, psi1
    phi = conditional_round(Thetas[0])
    psi2 = conditional_round(Thetas[1])
    theta = conditional_round(Thetas[2])
    psi1 = conditional_round(Thetas[3])
    
    
    global zz, ww2, uu, ww1
    
    zz = 1 / tan(phi / 2)
    ww2 = 1 / tan(psi2 / 2)
    uu = 1 / tan(theta / 2)
    ww1 = 1 / tan(psi1 / 2)
    
    global A, B, C
    A, B, C = coefficients()
    
    dirty_solutions = get_solutions_equations_squared(300)
    print(len(dirty_solutions))
    filtered_solutions, garbage_solutions = get_clean_solutions(dirty_solutions)
    text_to_write = ""
    for el in filtered_solutions:
        text_to_write += el[0]
        # print(el[0])

    text_to_write += "\nSOLUTIONS WITH SMALL ANGLES\n\n"
    # print("\nSOLUTIONS WITH SMALL ANGLES\n\n")
    for el in garbage_solutions:
        text_to_write += el[0]
        # print(el[0])
        
    with open("a_solutions.txt", "w") as cleanDataFile:
        cleanDataFile.write(text_to_write)
        print(f"File 'a_solutions.txt' created successfully.")
        
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Elapsed time: {elapsed_time:.2f} minutes")
    
if __name__ == "__main__":
    main()