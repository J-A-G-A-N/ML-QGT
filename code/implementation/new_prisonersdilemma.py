import numpy as np
from numba import njit

@njit
def entangler(gamma:np.float64)->np.ndarray:
    c = np.cos(gamma / 2.0)
    s = np.sin(gamma / 2.0)

    J = np.zeros((4,4),dtype=np.complex128)

    J[0,0] = c
    J[0,3] = 1j * s
    
    J[1,1] = c
    J[1,2] = 1j * s

    J[2,1] = 1j * s
    J[2,2] = c

    J[3,0] =  1j * s
    J[3,3] = c

    return J

@njit
def dagger(A):
    result = np.empty(
        (A.shape[1], A.shape[0]),
        dtype=np.complex128
    )

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            result[j, i] = np.conjugate(A[i, j])

    return result

@njit
def kron(A, B):
    result = np.empty(
        (A.shape[0] * B.shape[0],
         A.shape[1] * B.shape[1]),
        dtype=np.complex128
    )

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(B.shape[0]):
                for l in range(B.shape[1]):
                    result[i * B.shape[0] + k,
                           j * B.shape[1] + l] = A[i, j] * B[k, l]

    return result



# @njit
# def RX(angle:np.float64)->np.ndarray:
#     rx = np.zeros((2,2),dtype=np.complex128)

#     c = np.cos(angle/2)
#     s = np.sin(angle/2)

#     rx[0,0] = c
#     rx[0,1] = -1j * s
#     rx[1,0] = -1j * s
#     rx[1,1] = c

#     return rx

# @njit
# def RY(angle:np.float64)->np.ndarray:
#     ry = np.zeros((2,2),dtype=np.complex128)

#     c = np.cos(angle/2.0)
#     s = np.sin(angle/2.0)

#     ry[0, 0] = c
#     ry[0, 1] = -s
#     ry[1, 0] = s
#     ry[1, 1] = c

#     return ry


# @njit
# def strategy(params:np.ndarray)->np.ndarray:
#     return RX(params[0]) @ RY(params[1]) @ RX(params[2])    


# @njit
# def su2_to_xyx(U):
#     w = np.real(U[0, 0])
#     z = -np.imag(U[0, 0])

#     x = -np.imag(U[1, 0])
#     y = np.real(U[1, 0])

#     A = np.arctan2(x, w)
#     B = np.arctan2(z, y)

#     alpha = 2.0 * np.arctan2(
#         np.sqrt(y*y + z*z),
#         np.sqrt(w*w + x*x)
#     )

#     theta = A + B
#     beta = A - B

#     theta = np.mod(theta, 2.0 * np.pi)
#     alpha = np.mod(alpha, 2.0 * np.pi)
#     beta = np.mod(beta, 2.0 * np.pi)

#     return np.array([theta, alpha, beta])


@njit
def exp_su2(direction):

    dx = direction[0]
    dy = direction[1]
    dz = direction[2]


    r = np.sqrt(dx*dx + dy*dy + dz*dz)

    U = np.zeros((2,2), dtype=np.complex128)

    if r <1e-15:
        U[0, 0] = 1.0 
        U[1, 1] = 1.0 
        return U

    c = np.cos(r / 2.0)
    s = np.sin(r / 2.0) / r

    U[0, 0] = c -1j * s *dz
    U[0, 1] = -1j * s * dx - s *dy
    U[1, 0] = -1j * s * dx + s *dy
    U[1, 1] = c + 1j * s * dz

    return U


@njit
def strategy(params: np.ndarray) -> np.ndarray:
    theta = params[0]
    alpha = params[1]
    beta = params[2]

    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)

    U = np.zeros((2, 2), dtype=np.complex128)

    U[0, 0] = np.exp(1j * alpha) * c
    U[0, 1] = 1j * np.exp(1j * beta) * s

    U[1, 0] = 1j * np.exp(-1j * beta) * s
    U[1, 1] = np.exp(-1j * alpha) * c

    return U


@njit
def compute_payoffs(
    gamma: np.float64,
    alice_U: np.ndarray,
    bob_U: np.ndarray,
    p:int,
) -> tuple[np.float64, np.float64]:

    J = entangler(gamma)
    J_dagger = dagger(J)

    entangled_state = J @ np.array(
        [1, 0, 0, 0],
        dtype=np.complex128
    )

    if p == 0:
        print("Entangler:")
        print(J)


    # evolved_state = (
    #     kron(alice_U, bob_U)
    #     @ entangled_state
    # )

    evolved_state = entangled_state.reshape(2,2)
    evolved_state = alice_U @ evolved_state@ bob_U.T

    final_state = J_dagger @ evolved_state.reshape(4)

    probs = np.abs(final_state) ** 2

    R = 3.0
    T = 5.0
    S = 0.0
    P = 1.0

    alice_payoff = (
        R * probs[0] +
        S * probs[1] +
        T * probs[2] +
        P * probs[3]
    )

    bob_payoff = (
        R * probs[0] +
        T * probs[1] +
        S * probs[2] +
        P * probs[3]
    )


    return alice_payoff, bob_payoff
def check_unitary(U):
    I = np.eye(2, dtype=np.complex128)

    unitary_error = np.linalg.norm(U.conj().T @ U - I)
    determinant_error = abs(np.linalg.det(U) - 1.0)

    return unitary_error, determinant_error


@njit
def su2_to_theta_alpha_beta(U: np.ndarray) -> np.ndarray:
    a = U[0, 0]
    b = U[0, 1]

    abs_a = np.abs(a)
    abs_b = np.abs(b)

    theta = 2.0 * np.arctan2(abs_b, abs_a)

    alpha = np.arctan2(
        np.imag(a),
        np.real(a)
    )

    beta = np.arctan2(
        np.imag(b),
        np.real(b)
    ) - np.pi / 2.0

    if alpha > np.pi:
        alpha -= 2.0 * np.pi
    elif alpha <= -np.pi:
        alpha += 2.0 * np.pi

    if beta > np.pi:
        beta -= 2.0 * np.pi
    elif beta <= -np.pi:
        beta += 2.0 * np.pi

    return np.array([
        theta,
        alpha,
        beta
     ])



def main():
    gamma = np.pi/2
    U_D = strategy([np.pi, 0.0, 0.0])
    U_C = strategy([0.0, 0.0, 0.0])
    U_Q = strategy([0.0, np.pi/2, np.pi/2])
    print("CC:", compute_payoffs(gamma, U_C,U_C,1))
    print("CD:", compute_payoffs(gamma, U_C,U_D,1))
    print("DC:", compute_payoffs(gamma, U_D,U_C,1))
    print("DD:", compute_payoffs(gamma, U_D,U_D,1))

    print("CQ:", compute_payoffs(gamma, U_C,U_Q,1))
    print("QD:", compute_payoffs(gamma, U_Q,U_D,1))
    print("DQ:", compute_payoffs(gamma, U_D,U_Q,1))
    print("DQ:", compute_payoffs(gamma, U_D,U_Q,1))
    print("QQ:", compute_payoffs(gamma, U_Q,U_Q,1))



if __name__ == "__main__":
    main()


# U_C = np.eye(2, dtype=np.complex128)
# U_D = RX(np.pi)
# gamma = 0.0

# print("CC:", compute_payoffs(gamma, U_C, U_C, 0))
# print("CD:", compute_payoffs(gamma, U_C, U_D, 0))
# print("DC:", compute_payoffs(gamma, U_D, U_C, 0))
# print("DD:", compute_payoffs(gamma, U_D, U_D, 0))
# # alice_prams = 2.0 * np.pi * np.random.rand(3)
# # bob_params = 2.0 * np.pi * np.random.rand(3)
# # UA = strategy(alice_prams)
# # UB = strategy(bob_params)

# # print(compute_payoffs(0,UA,UB))


# # params = 
# #
# # U = strategy(params)
# #
# # v = np.random.randn(3)
# #
# # U_new = U @ exp_su2(v)
# #
# # print(check_unitary(U_new))
# # #
# # params = 2.0 * np.pi * np.random.rand(3)
# #
# # U = strategy(params)
# #
# # print(U)
# #
# # print(check_unitary(U))



# # v = np.random.randn(3)
# #
# # V = exp_su2(v)
# #
# # print(check_unitary(V))



# import numpy as np
# from numba import njit


# @njit
# def entangler(gamma: np.float64) -> np.ndarray:
#     c = np.cos(gamma / 2.0)
#     s = np.sin(gamma / 2.0)

#     J = np.zeros((4, 4), dtype=np.complex128)

#     J[0, 0] = c
#     J[0, 3] = 1j * s

#     J[1, 1] = c
#     J[1, 2] = 1j * s

#     J[2, 1] = 1j * s
#     J[2, 2] = c

#     J[3, 0] = 1j * s
#     J[3, 3] = c

#     return J




# @njit
# def exp_su2(direction: np.ndarray) -> np.ndarray:
#     dx = direction[0]
#     dy = direction[1]
#     dz = direction[2]

#     r = np.sqrt(dx * dx + dy * dy + dz * dz)

#     U = np.zeros((2, 2), dtype=np.complex128)

#     if r < 1e-15:
#         U[0, 0] = 1.0
#         U[1, 1] = 1.0
#         return U

#     c = np.cos(r / 2.0)
#     s = np.sin(r / 2.0) / r

#     U[0, 0] = c - 1j * s * dz
#     U[0, 1] = -1j * s * dx - s * dy
#     U[1, 0] = -1j * s * dx + s * dy
#     U[1, 1] = c + 1j * s * dz

#     return U


# @njit
# def dagger(A: np.ndarray) -> np.ndarray:
#     result = np.empty(
#         (A.shape[1], A.shape[0]),
#         dtype=np.complex128
#     )

#     for i in range(A.shape[0]):
#         for j in range(A.shape[1]):
#             result[j, i] = np.conjugate(A[i, j])

#     return result


# @njit
# def kron(A: np.ndarray, B: np.ndarray) -> np.ndarray:
#     result = np.empty(
#         (
#             A.shape[0] * B.shape[0],
#             A.shape[1] * B.shape[1]
#         ),
#         dtype=np.complex128
#     )

#     for i in range(A.shape[0]):
#         for j in range(A.shape[1]):
#             for k in range(B.shape[0]):
#                 for l in range(B.shape[1]):
#                     result[
#                         i * B.shape[0] + k,
#                         j * B.shape[1] + l
#                     ] = A[i, j] * B[k, l]

#     return result


# @njit
# def compute_payoffs(
#     gamma: np.float64,
#     alice_U: np.ndarray,
#     bob_U: np.ndarray,
#     p: int,
# ) -> tuple[np.float64, np.float64]:

#     J = entangler(gamma)
#     J_dagger = dagger(J)

#     initial_state = np.array(
#         [1, 0, 0, 0],
#         dtype=np.complex128
#     )

#     entangled_state = J @ initial_state

#     if p == 0:
#         print("Entangler:")
#         print(J)

#     evolved_state = (
#         kron(alice_U, bob_U)
#         @ entangled_state
#     )

#     final_state = J_dagger @ evolved_state

#     probs = np.abs(final_state) ** 2

#     R = 3.0
#     T = 5.0
#     S = 0.0
#     P = 1.0

#     alice_payoff = (
#         R * probs[0]
#         + S * probs[1]
#         + T * probs[2]
#         + P * probs[3]
#     )

#     bob_payoff = (
#         R * probs[0]
#         + T * probs[1]
#         + S * probs[2]
#         + P * probs[3]
#     )

#     return alice_payoff, bob_payoff


# def check_unitary(U):
#     I = np.eye(2, dtype=np.complex128)

#     unitary_error = np.linalg.norm(
#         U.conj().T @ U - I
#     )

#     determinant_error = abs(
#         np.linalg.det(U) - 1.0
#     )

#     return unitary_error, determinant_error


