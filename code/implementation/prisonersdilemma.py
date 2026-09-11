import numpy as np


class Player:
    def __init__(self) -> None:
        self.theta:float = 0
        self.alpha:float = 0
        self.beta:float = 0
    def set(self,theta:float,alpha:float,beta:float):
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
    def get(self)->tuple[float,float,float]:
        return self.theta, self.alpha, self.beta
    
class Operator:
    @staticmethod
    def I(): return np.eye(2,dtype=complex)
    @staticmethod
    def pauli_x(): return np.array([[0,1],[1,0]])

    @classmethod
    def entangement_operator(cls,gamma:float):
        X = cls.pauli_x()
        I = cls.I()
        XX = np.kron(X,X)

        return  (np.cos(gamma/2) * np.kron(I,I)
        + 1j * np.sin(gamma/2) * XX
        )

    @staticmethod
    def dagger(J:np.ndarray):
        return J.conj().T

    @classmethod
    def unitary(cls,player:Player):
        angle = player.theta / 2
        A = np.exp(1j * player.alpha) * np.cos(angle)
        B = 1j * np.exp(1j * player.beta) * np.sin(angle)
        C = 1j * np.exp(- 1j * player.beta) * np.sin(angle)
        D = np.exp(-1j * player.alpha) * np.cos(angle)
        return np.array([[A,B],[C,D]])



# construct the J operator
# contruct the J dagger operator
# consturt the players U operators
# apply opertors to the inital state

class PrisonersDilemma:

    def __init__(self,gamma):
        self.ket_00 = np.array([1,0,0,0])
        self.J = Operator.entangement_operator(gamma)
        self.J_dagger = Operator.dagger(self.J)
        self.entangle_state =self.J @ self.ket_00

    def run(self,alice:Player,bob:Player):
        UA = Operator.unitary(alice)
        UB = Operator.unitary(bob)

        state = self.entangle_state.reshape(2,2)

        state = UA @ state @ UB.T
        evolved_state = state.reshape(4)

        return self.J_dagger @ evolved_state


    @staticmethod
    def compute_probabilities(state):
        return np.abs(state) ** 2

    @staticmethod
    def compute_payoffs(probs):
        P00, P01, P10, P11 = probs
        Reward = 3 
        Sucker_s_payoff = 0 
        Temptation = 5
        Punishment = 1

        alice_payoff = (
                Reward * P00 +
                Sucker_s_payoff * P01 +
                Temptation * P10 +
                Punishment * P11
        ) 

        bob_payoff = (
                Reward * P00 +
                Temptation * P01 +
                Sucker_s_payoff * P10 +
                Punishment * P11
        ) 
        return alice_payoff, bob_payoff
