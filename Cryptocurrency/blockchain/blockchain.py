from blockchain.block import Block
from blockchain.transaction import Transaction
from decimal import Decimal
import time as t

class Blockchain:
    """
    The Blockchain stores Blocks and points on the latest added Block.
    """
    def __init__(self):
        """
        Constructor
        """
        self.version = 0
        self.chain_length = 0
        previous_hash = '0' * 64
        self.transactions = []
        self.pending_transactions = []
        nonce = 0
        self.DIFFICULTY_ADJUST = 10
        self.reward = 800.0
        self.fees = Decimal(0.0)
        self.difficulty = 1.0
        self.target = 2 ** 237 - self.difficulty 
        print(self.target)
        self.chain = [Block(self.chain_length,self.version,previous_hash,self.difficulty,nonce,self.transactions,0.0)]

    def add(self,nonce,miner_wallet_id):
        previous_hash = self.chain[-1].own_hash
        n = 0
        while 2 ** n-1 <= len(self.transactions) + len(self.pending_transactions):
            n += 1
        n -= 1

        i = len(self.pending_transactions)
        # print('-------------------------------------------------')
        while i < 2**n-1:
            self.fees += self.transactions[0].fee
            self.pending_transactions.append(self.transactions[0])
            self.transactions.pop(0)
            # print('i: ' ,i)
            i += 1
        # for transaction in self.pending_transactions:
            # fees += transaction.fee
        # print('Fees: ', fees, 'Marecoin')
        mining_reward = Transaction(0,miner_wallet_id,self.reward + float(self.fees),True)
        self.pending_transactions.insert(0,mining_reward)
        
        # print(len(self.pending_transactions))
        new_block = Block(self.chain_length+1,self.version,previous_hash,self.difficulty,nonce,self.pending_transactions,self.reward) 
        int_hash = int(new_block.own_hash, 16)
        if int_hash < self.target:
            self.chain_length += 1
            self.chain.append(new_block)
            self.pending_transactions = [] 
            self.fees = Decimal(0.0)  
            print()
            print("Block Mined with hash: " + new_block.own_hash)
            print("Difficulty: ", self.difficulty)
            print("Target: ", self.target)
            print()
            self.set_target()
            self.halving()
            return True
        # print(new_block.own_hash)
        self.pending_transactions.pop(0)
        return False
       

    def halving(self):
        if self.chain_length % 672000 == 0:
            self.reward /= 2
    
    def set_target(self):
        if self.chain_length % self.DIFFICULTY_ADJUST == 0:
            start_checking = self.chain_length - self.DIFFICULTY_ADJUST
            time = 0
            while start_checking < self.chain_length:
                print("s: " ,start_checking)
                print("c: " ,len(self.chain))
                start_time = self.chain[start_checking].blockheader.timeStamp
                end_time = self.chain[start_checking+1].blockheader.timeStamp
                difference = end_time - start_time
                time += difference.microseconds
                start_checking += 1
            time_taken = time // self.DIFFICULTY_ADJUST
            factor = 1800000000 / time_taken
            self.difficulty = round(self.difficulty * factor,2)
            if self.difficulty < 1.0:
                self.difficulty = 1.0 
            print('Difficulty: ', self.difficulty) 
            t.sleep(2)   
            self.target = 2**237 - self.difficulty

    def add_transaction(self,transaction):
        self.transactions.append(transaction)
        

