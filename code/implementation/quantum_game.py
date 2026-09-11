import numpy as np
import qiskit
import qiskit_aer
from enum import Enum
from numpy.typing import NDArray

class ExecutionType(Enum):
    state_vector  = 1 
    aer_sim  = 2 

class Player:
    def __init__(self,theta:float=0.0,alpha:float=0.0,beta:float=0.0):

        self.theta:float = theta
        self.alpha:float = alpha
        self.beta:float = beta

    def get_parameters(self) -> tuple[float,float,float]:
        return self.theta, self.alpha, self.beta,


class UnitaryOperator:
    @staticmethod
    def create(theta:float = 0.0, alpha:float = 0.0, beta:float = 0.0) -> NDArray[np.complex128]:
        A = np.exp(1j* alpha) * np.cos(theta / 2) 
        B = 1j * np.exp(1j * beta) *  np.sin(theta / 2)
        C = 1j * np.exp(- 1j * beta) *  np.sin(theta / 2)
        D = np.exp( - 1j* alpha) * np.cos(theta / 2) 

        U = np.array(([A,B],[C,D]),dtype = np.complex128)

        return U

class EntanglingOperator:
    @staticmethod
    def pauli_x() -> NDArray[np.complex128]:
        return np.array(([0,1],[1,0]),dtype=np.complex128)

    @staticmethod
    def identity()->NDArray[np.complex128]:
        return np.eye(2,dtype=np.complex128)

    @staticmethod
    def tensor_product(A,B)->NDArray[np.complex128]:
        return np.kron(A,B)

    @staticmethod
    def dagger(J)->NDArray[np.complex128]:
        return J.conj().T

    @classmethod
    def create(cls,gamma)->NDArray[np.complex128]:
        X = cls.pauli_x()
        I = cls.identity()
        XX = cls.tensor_product(X,X)

        J = (np.cos(gamma/2) * cls.tensor_product(I,I)
        + 1j * np.sin(gamma/2) * XX
        )

        return J


class QuantumPrisonersDilemma:
    def __init__(self,gamma):
        self.gamma = gamma
        self.J = EntanglingOperator.create(gamma)
        self.J_dagger = EntanglingOperator.dagger(self.J)
        self.execution_type = 0
        self.state_vector = None
    def get_player_unitary(self,player:Player):

        theta, alpha, beta  = player.get_parameters()
        return UnitaryOperator.create(theta,alpha,beta)

    def create_circuit(self,player_a:Player,player_b:Player):
        self.qc = qiskit.QuantumCircuit(2)
        Operator = qiskit.quantum_info.Operator

        self.qc.unitary(Operator(self.J), [0,1],label=f"$J$")

        UA = self.get_player_unitary(player_a)
        self.qc.unitary(Operator(UA),[0],label=r"$U_A$")

        UB = self.get_player_unitary(player_b)
        self.qc.unitary(Operator(UB),[1],label=r"$U_B$")

        self.qc.unitary(Operator(self.J_dagger),[0,1],label=r"$J^\dagger$")

    def simulate(self,execution_type:ExecutionType,shots:int=1024):

        self.execution_type =execution_type 
        match execution_type:
            case ExecutionType.aer_sim:
                if shots is None or shots <=0:
                    raise ValueError("'shots' should have value greater than 0.")

                self.shots = shots
                meas_circuit = self.qc.copy()
                meas_circuit.measure_all()

                simulator = qiskit_aer.AerSimulator()
                results = simulator.run(meas_circuit ,shots = shots).result()
                self.counts = results.get_counts()

            case ExecutionType.state_vector:
                self.state_vector =  qiskit.quantum_info.Statevector.from_instruction(self.qc)

# this is because of the qiskit qubit ordering
# qiskit_probabilities[2] → actual probabilities[1]
# qiskit_probabilities[1] → actual probabilities[2]
    def compute_probabilities(self):
        match self.execution_type:
            case ExecutionType.aer_sim:
                probabilities = np.array([
                    self.counts.get("00",0) / self.shots, 
                    self.counts.get("10",0) / self.shots,
                    self.counts.get("01",0) / self.shots,
                    self.counts.get("11",0) / self.shots
                    ])

                if not np.isclose(np.sum(probabilities),1):
                    raise ValueError("'probabilities' do not add upto 1")
            case ExecutionType.state_vector:
                qiskit_probabilities = self.state_vector.probabilities()
                probabilities = np.array([
                    qiskit_probabilities[0],
                    qiskit_probabilities[2],
                    qiskit_probabilities[1],
                    qiskit_probabilities[3]
                    ])
            case _:
                raise ValueError(f"Unknown execution type: {self.execution_type}")

        return probabilities
    def compute_payoff(self,probabilities)->tuple[float,float]:
        P00, P01, P10, P11 = probabilities

        R = 3 # Reward 
        T = 5 # Temptation
        S = 0 # Sucker's payoff
        P = 1 # Punishment


        player_a_payoff = (
           R * P00 +
           S * P01 +
           T * P10 +
           P * P11 
        )

        
        player_b_payoff =(
           R * P00 +
           T * P01 +
           S * P10 +
           P * P11 
        ) 
        return player_a_payoff, player_b_payoff


    def get_counts(self): return self.counts

    def get_circuit(self): return self.qc


