#########################################################################################################################################################################
##             Problem                                                                                                                                                  #
## Play a n-k coh coh game where user should not place consecutive k same pieces in given row, column, and diagonols                                                     #
## For this problem alpha-beta pruning algorithm is used .                                                                                                             #
##
##             Heuristic Function
## A heuristic function is designed as a reverse of normal tic tac toe game but here the board score is based on the difference of the no of k length rows opened
## for player and opponent. If a terminal state is reached (Win for player) -10^k is assigned to the board and 10^k is assigned to the board for opponent.
##
##             Algorithm                                                                                                                                                   #
## A normal minimax is used with alpha beta pruning algorithm but a slight modification is done. For player white Max is used and for Black Min is used as this is        #
## reverse of tic tac toe .                                                                                                                                                #
## #########################################################################################################################################################################
import numpy as np
import random
import sys


class Game():

    def __init__(self,n,k,board,user_time):
        self.N = n
        self.k = k
        self.depth = 2
        self.player = None
        self.marker=None
        self.initBoard = self.create_board(board)
        self.maximizing=False
        self.user_time=user_time


    def __repr__(self):
        return self.nextmove()

    def create_board(self,board):
        init_board=[ls for ls in [[i for i in board[i:i + len(board)/self.N]] for i in range(0, len(board), len(board)/self.N)]]
        return init_board

    '''This function used to add piece'''
    def addpiece(self,board,row,col):
        self.marker = self.findNextPlayer(board)
        return board[0:row] + [board[row][0:col] + [self.marker,] + board[row][col+1:]] + board[row+1:]

    '''This function used to remove duplicates of the child boards'''
    def removeduplicateboard(self,all_boards):
        boards=[np.matrix(state) for i,state in enumerate(all_boards)]
        lists=boards

        for i in range(0, len(boards)):
            for j in range(1, len(boards)):
                if (np.all(np.fliplr(boards[i]) == boards[j]) or np.all(np.flipud(boards[i]) == boards[j])) and i!=j:
                    lists = lists[:j] + lists[j + 1:]


        return [i.tolist() for i in lists]

    # Getting child board of the current board
    def successors(self,board):

        return [self.addpiece(board, r, c) for r in range(0,self.N) for c in range(0,self.N) if board[r][c] == '.']

    '''Functiton for finding the current player of the board'''
    def findNextPlayer(self,board):

        p1 = [j for i in board for chr in i for j in chr if j== 'w']
        p2 = [j for i in board for chr in i for j in chr if j== 'b']
        player = 'b' if len(p1) > len(p2) else 'w'
        return player

    ##########################################################################################
    #For evaluating a board the seperate counts of black pieces and white pieces where      ##
    # it can and returning the difference . If a terminal node is reached a higher score    ##
    # is given for the board. If w is the current player and the board is terminal that     ##
    # is if w looses then a bigger negative will be given such that mini-max assumes        ##
    # it is a bad board for 'w'alpha beta prunning wil not take the board                   ##
    #A heuristic value of negative is not good board for w and vice versa                   ##
    ##########################################################################################

    def game_evaluation(self,board):
        w_count, b_count = 0,0
        temp_board = np.matrix(board)
        diags = [temp_board[::-1, :].diagonal(i) for i in range(-3, 4)]
        diags.extend(temp_board.diagonal(i) for i in range(3, -4, -1))


        #Getting the k size row opened for pieces
        for row in board:
            for r in range((self.N-self.k) + 1):
                if 'b' not in row[r:r+self.k]:
                    if 'w' in row[r:r+self.k] or '.' in row[r:r+self.k]:
                        w_count += 1
                        if ''.join(row[r:r + self.k]) == 'w' * self.k:
                            return -10 ** self.k
                if 'w' not in row[r:r+self.k]:
                    if 'b' in row[r:r+self.k] or '.' in row[r:r+self.k]:
                        b_count += 1
                        if ''.join(row[r:r + self.k]) == 'b' * self.k:
                            return 10 ** self.k

        #Formatting columns
        col_string = ''
        col_list,final_col=[],[]
        col_list=[temp_board[:,i].tolist() for i in range(0, self.N)]

        for c in col_list:
            col_string=''
            for d in c:
                col_string += ''.join(d)
            final_col.append(col_string)

        # Getting the k size row opened for pieces in columns

        for col in final_col:
            for r in range((self.N - self.k) + 1):
                if 'b' not in col[r:r + self.k]:
                    if 'w' in col[r:r + self.k] or '.' in col[r:r + self.k]:
                        w_count+=1
                        if ''.join(col[r:r + self.k]) == 'w' * self.k:
                            return -10 ** self.k

                if 'w' not in col[r:r + self.k]:
                    if 'b' in col[r:r + self.k] or '.' in col[r:r + self.k]:
                        b_count += 1
                        if ''.join(col[r:r + self.k]) == 'b' * self.k:
                            return 10 ** self.k

        # Getting the k size row opened for pieces in diagnols
        for n in diags:
            for i in n.tolist():
                if len(i) >= self.k:
                    for r in range(len(i)-1):
                        if r+self.k <= len(i):
                            if 'b' not in i[r:r + self.k]:
                                if 'w' in i[r:r + self.k] or '.' in i[r:r + self.k]:
                                    w_count += 1
                                    if ''.join(i[r:r+self.k]) == 'w'*self.k:
                                        return -10**self.k

                            if 'w' not in i[r:r + self.k]:
                                if 'b' in i[r:r + self.k] or '.' in i[r:r + self.k]:
                                    b_count += 1
                                    if ''.join(i[r:r+self.k]) == 'b'*self.k:
                                        return 10**self.k

        ###########################################################################
        ## Difference of w_count and h_count will be taken as a heuristic value  ##
        ###########################################################################
        h_val = b_count - w_count
        return h_val


    #####################################################################
    ## Return all the rows and column and diagnols                     ##
    #####################################################################
    def get_allrows(self,board):
        temp_board = np.matrix(board)
        diags = [temp_board[::-1, :].diagonal(i) for i in range(-3, 4)]
        diags.extend(temp_board.diagonal(i) for i in range(3, -4, -1))

        row = [row.tolist() for row in temp_board]

        column = [temp_board[:, i].tolist() for i in range(0, self.N)]
        diagnols = [i for n in diags for i in n.tolist() if len(i) >= self.k]

        return row,column,diagnols

    ####################################################################
    #function for checking a terminal state is reached or not that is  #
    #win for w or win for L of draw                                    #
    ####################################################################
    def terminal_state(self,board):
        return_val=False
        #getting the indicidaul rows and columns and diagonals of the board
        row,column,diagnols = self.get_allrows(board)

        #checking the board is in draw position
        row_str = ''
        for row in board:
            for r in row:
                row_str += ''.join(r)
        if '.' not in row_str:
            return_val = True


        #checking a loose is occured in any row of the board (consecutive k pieces is present)
        for row in board:
            if ''.join('w'*self.k) in ''.join(row) or ''.join('b'*self.k) in ''.join(row):
                return_val = True

        # checking a loose is occured in any column of the board (consecutive k pieces is present)
        col_string = ''
        for col in column:
            col_string=''
            for col_cell in col:
                col_string += ''.join(col_cell[0])
            if ''.join('w'*self.k) in col_string or ''.join('b'*self.k) in col_string:
                return_val = True

        #checking a loose is occurred in any diagonals (consecutive k pieces is present)
        for d in diagnols:
            if ''.join('w'*self.k) in ''.join(d) or ''.join('b'*self.k) in ''.join(d):
                return_val = True

        #returning the value as True if a terminal condition (win or draw is in board)
        return return_val

    #######################################################
    #Function to determine the next move of the board    ##
    #######################################################
    def nextmove(self):
        ###################################################
        #check whether the given board is terminal or not #
        ###################################################
        if self.terminal_state(self.initBoard):
            return 'Terminal State is reached.'
        else:
            prev_state=None
            prev_score=None
            current_state=None
            root_board=self.initBoard
            self.player = self.findNextPlayer(root_board)
            self.maximizing=True if self.player == 'w' else False

            ############################################################################################
            #IDS is initialised here                                                                 ###
            # This will iterate over game tree to a maximum of 9999 times of the actual depth of tree ###
            # by exploring incremented  game tree depth                                              ###
            ############################################################################################
            for depth in range(1,9999):
                ######################################################################################
                ## Alpha beta Prunning is called for the root node since the heuristic is evaluated ##
                # for winning scenario the maximising is called first                               ##
                #                                                                                 ####
                ######################################################################################

                score,returned_state=self.alpha_beta(root_board,depth,self.maximizing,float('-inf'),float('inf'))

                #########################################################################################
                # If returned board from alph beta prunning is terminal and the best board(non-terminal)#
                #  from previous                                                                        #
                # iteration is used as best board the depth iteration is stopeed                        #
                #########################################################################################

                if prev_state == None:
                    prev_state = returned_state
                    prev_state = self.game_evaluation(prev_state)
                current_state = returned_state
                current_score = self.game_evaluation(current_state)

                if self.player == 'w' and current_score <= -10**self.k:
                    current_state = prev_state
                else:
                    prev_state = current_state
                    prev_score = current_score

                if self.player == 'b' and current_score >= 10**self.k:
                    current_state=prev_state
                else:
                    prev_state = current_state
                    prev_score = current_score
                ###########################################################################################
                #                                                                                        ##
                #The out put is printed here                                                             ##
                #                                                                                        ##
                ###########################################################################################
                print ' '.join(str(li2) for li1 in current_state for li2 in li1)

    #######################################
    #alpha beta prunning functions      ###
    #######################################

    def alpha_beta(self,board,depth,maximising,alpha,beta):
        #################################################
        ## Checking depth is reached and return the     #
        # heuristic value                               #
        #################################################
        if depth == 0 or self.terminal_state(board):
            score=self.game_evaluation(board)
            return score,board
        else:
            if maximising == True:
                best_move=None
                for state in self.removeduplicateboard(self.successors(board)):
                    temp_alpha=alpha
                    t_alpha,child=self.alpha_beta(state,depth-1,True, alpha, beta)
                    alpha = max(alpha, t_alpha)
                    if alpha != temp_alpha:best_move=state
                    if beta <= alpha:
                        return alpha, best_move
                #Returning alpha score and board
                return alpha,best_move
            if maximising==False:
                best_move = None
                for state in self.removeduplicateboard(self.successors(board)):
                    temp_beta=beta
                    t_beta,child=self.alpha_beta(state,depth-1,False, alpha, beta)
                    beta = min(beta,t_beta)
                    if beta != temp_beta:
                        best_move = state
                    if beta <= alpha:
                        return beta, best_move
                #returning beta score and board
                return beta,best_move


def main():
    n, k, board, user_time=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3].lower(),int(sys.argv[4])

    ###################################################################
    #For debugging
    #uncomment this
    ###################################################################
    #n,k,board,user_time=3,3,'wbwb.wb..',5
    # Check the board is valid
    if n * n != len(board):
        print 'Please enter a valid board'
    else:

        game=Game(n,k,board,user_time)
        print(game)

if __name__ == '__main__':
    main()

''