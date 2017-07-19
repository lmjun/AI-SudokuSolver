#!/usr/bin/python

import numpy as np
import sys


'''
File reading function for sudoku
Input: filename
Output: a list of 2D numpy matrices representing sudokus
'''
def read_sudoku(fname):
    with open(fname) as f:
        f_input = [x.strip('\r\n') for x in f.readlines()]

    sudoku_list = []
    for i in xrange(len(f_input)):
        sudoku = np.zeros((9, 9))
        temp = f_input[i]
        for j in xrange(0, len(temp), 9):
            sudoku_row = temp[j:j + 9]
            for k in xrange(0, 9):
                sudoku[j / 9][k] = sudoku_row[k]
        sudoku_list.append(sudoku)

    return sudoku_list


'''
Printing function for sudoku,
Input: a 2D numpy matrix
'''
def print_sudoku(sudoku):
    print '+-------+-------+-------+'
    for i in xrange(0, 9):
        for j in xrange(0, 9):
            if j == 0:
                print '|',
            if sudoku[i][j] != 0:
                print int(sudoku[i][j]),
            else:
                print '*',
            if (j + 1) % 3 == 0:
                print '|',
        print ''
        if (i + 1) % 3 == 0:
            print '+-------+-------+-------+'
    print ''


'''
Utility function for finding constraints
Input: coordinate [row, col]
Output: constraints
'''

def get_constraint(coordinate, sudoku):
    value = sudoku[coordinate[0]][coordinate[1]]
    if value == 0:
        row = sudoku[coordinate[0], :]
        col = sudoku[:, coordinate[1]]

        row_constraint = row[np.nonzero(row)]
        col_constraint = col[np.nonzero(col)]
        block = get_block(coordinate, sudoku)

        blo_constraint = block[np.nonzero(block)]

        all_constraint = np.unique(np.concatenate((row_constraint, col_constraint, blo_constraint)))
        return all_constraint
    else:
        print 'not a variable'


'''
Utility function for getting a 3x3 sudoku block given a coordinate
Input: coordinate [row, col]
Output: 3x3 numpy matrix
'''
def get_block(coordinate, sudoku):
    row_range = [3 * (coordinate[1] / 3), 3 * (coordinate[1] / 3) + 3]
    col_range = [3 * (coordinate[0] / 3), 3 * (coordinate[0] / 3) + 3]

    return sudoku[col_range[0]:col_range[1], row_range[0]:row_range[1]]

def get_domain(constraints):
	domain = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
	for n in range (constraints.shape[0] - 1, -1, -1):	
		domain =  np.delete(domain, constraints[n] - 1) 
	return domain

def revise(X, Y, soln, domains):
	revised = False
	
	#each possible value of X
	for it in range(0, 9):
		if(domains[X[0]][X[1]][it] == 1):
			#model soln, check if Y domain size > 0
			soln[X[0]][X[1]] = it + 1 
			dom = get_domain(get_constraint([Y[0], Y[1]], soln))
			soln[X[0]][X[1]] = 0
			
			if(dom.shape[0] == 0):
				domains[X[0]][X[1]][it] = 0
				revised = True
	return revised

def isNeighbor(X, Y):
	#check if not arc to self
	if(X[0] == Y[0] and X[1] == Y[1]):
		return False
	#check if arc to same col or row
	if(X[0] == Y[0] or X[1] == Y[1]):
		return True
	#check if arc in same block
	cR = [3 * (X[1] / 3), 3 * (X[1] / 3) + 3]
	rR = [3 * (X[0] / 3), 3 * (X[0] / 3) + 3]
	if(Y[1] >= cR[0] and Y[1] < cR[1] and Y[0] >= rR[0] and Y[0] < rR[1]):
		return True	
	return False

'''
AC-3 Algorithm
Input: 2D numpy matrix
Output: return True if a solution is found, with solved sudoku, False otherwise, with original sudoku
'''
def ac3(sudoku):
	#list of unassigned variables
	r0 = np.where(sudoku == 0)[0]
	c0 = np.where(sudoku == 0)[1]
	solved_sudoku = np.copy(sudoku)
	
	#domains[r][c][value]: 1 = possible, 0 = not possible
	domains = np.zeros((sudoku.shape[0], sudoku.shape[1], 9))

	#for each unassigned variable find domain
	for var in range (0, r0.shape[0]): 
		dom = get_domain(get_constraint([r0[var], c0[var]], solved_sudoku))
		for it in range(0, dom.shape[0]):
			domains[r0[var]][c0[var]][dom[it] - 1] = 1

	#first make queue of all arcs (neighboring unassigned domains)
	queue = np.zeros((0, 4))
	
	#for each unassigned variable
	for var in range (0, r0.shape[0]):
		#current unassigned variable coords [r0[var], c0[var]]
		#iterate through all 0s
		for arc in range(0, r0.shape[0]):	
			#if neighbors
			if(isNeighbor([r0[var], c0[var]],[r0[arc], c0[arc]])):
				array = np.array([r0[var], c0[var], r0[arc], c0[arc]])
				queue = np.vstack([queue, array])
			
	#while queue not empty
	while(queue.shape[0] > 0):
		Ir = queue[0][0]
		Ic = queue[0][1]
		Jr = queue[0][2]
		Jc = queue[0][3]
		#remove arc from queue
		queue = np.delete(queue, (0), axis=0)

		#make sure I and J are still unassigned
		if(solved_sudoku[Jr][Jc] == 0 and solved_sudoku[Ir][Ic] == 0):
			#check if D(J) got trimmed 
			if(revise([Ir, Ic], [Jr, Jc], solved_sudoku, domains)):
				#if all possible values of var are gone, no solution
				if 1 not in domains[Ir][Ic][:]:
					return False, solved_sudoku
				#add arcs between current and unassiged neighbors to queue
				for r in range(0, 9):
					for c in range(0, 9):
						if(isNeighbor([Ir, Ic], [r, c]) and solved_sudoku[r][c] == 0):
							array = np.array([Ir, Ic, r, c])
							queue = np.vstack([queue, array])	
	
				found = True
				while(found):
					found = False
					#check if I got assigned (if yes apply unary constraints)
					for r in range(0, 9):
						for c in range(0, 9):
							if (solved_sudoku[r][c] == 0):
								if(1 not in domains[r][c][:]):
									return False, solved_sudoku
								if(np.count_nonzero(domains[r][c][:]) == 1):
									found = True
									toR = r
									toC = c
					if(found):
						#assign				
						for it in range(0, 9):
							if(domains[toR][toC][it] == 1):
								solved_sudoku[toR][toC] = it + 1
						#update domains for unassigned neighbors of assigned variable
						for r in range(0, 9):
							for c in range(0, 9):
								if(solved_sudoku[r][c] == 0 and isNeighbor([toR, toC], [r, c])):
									constr = get_constraint([r, c], solved_sudoku)
									for it in range(0, constr.shape[0]):
										domains[r][c][constr[it] - 1] = 0
		
				
	return True, solved_sudoku		


def btsHelper(root, solved, solution):
	if(not solved):
		full = True
		
		#least constrained domain
		lcdR = lcdC = 0
		lcdMIN = 0
		
		#if any variables have domain size 0 prune branch
		for r in range(0, 9):
			for c in range(0, 9):
				if (root[r][c] == 0):
					full = False
					constr = get_constraint([r, c], root)
					if(constr.shape[0] == 9):
						return root
					if(constr.shape[0] > lcdMIN):
						lcdR = r
						lcdC = c
						lcdMIN = constr.shape[0]
						
		#if viable solution
		if(full):
			#won't start any more recursive calls
			solved = True
			solution = root
		#model newRoot with all possible values in domain of (lcdR,lcdC)
		else:
			newRoot = np.copy(root)
			dom = get_domain(get_constraint([lcdR, lcdC], root))
			#iterate through domain
			for it in range(0, dom.shape[0]):
				if(not solved):
					newRoot[lcdR][lcdC] = dom[it]
					#recursive call
					solution = btsHelper(newRoot, solved, solution)
					todo = False
					for r in range(0, 9):
						for c in range(0, 9):
							if (solution[r][c] == 0):
								todo = True
								break
						if(todo):
							break
					if(not todo):
						return solution
	return solution

'''
Backtracking search Algorithm
Input: 2D numpy matrix
Output: return True if a solution is found, with solved sudoku, False otherwise, with original sudoku
'''
def bts(sudoku):
	#apply ac3
	solved, solved_sudoku = ac3(sudoku)
	if(not solved):
		return False, solved_sudoku

	#check if complete soln
	todo = False
	for r in range(0, 9):
		for c in range(0, 9):
			if (solved_sudoku[r][c] == 0):
				todo = True
				break
		if(todo):
			break		
	if(todo):
		#only true if btshelper finds complete solution
		btsSolved = False
		solved_sudoku = (btsHelper(solved_sudoku, btsSolved, solved_sudoku))
		todo = False
		for r in range(0, 9):
			for c in range(0, 9):
				if (solved_sudoku[r][c] == 0):
					todo = True
					break
			if(todo):
				break
		if(not todo):
			btsSolved = True
			
		return btsSolved, solved_sudoku 
	else:
		return True, solved_sudoku
	
'''
Main function
'''
def main():
    sudoku_list = read_sudoku(sys.argv[1])
    solved_sudokus = []
    for sudoku in sudoku_list:
        print_sudoku(sudoku)
        if sys.argv[2] == 'ac3':
            print 'Using AC-3'
            solved, ret_sudoku = ac3(sudoku)
            if solved:
                print 'Solved Sudoku'
                print_sudoku(ret_sudoku)
            else:
                print 'No solution found'
            solved_sudokus.append(ret_sudoku.flatten())
        elif sys.argv[2] == 'bts':
            print 'Using backtracking search'
            solved, ret_sudoku = bts(sudoku)
            if solved:
                print 'Solved Sudoku'
                print_sudoku(ret_sudoku)
            else:
                print 'No solution found'
            solved_sudokus.append(ret_sudoku.flatten())
        else:
            print 'No such type'
        print ''

    np.savetxt('sudoku_solutions_'+sys.argv[2]+'.txt', solved_sudokus, fmt='%d', delimiter='')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Arguments error'
    else:
        main()
