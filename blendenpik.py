"""
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# DESCRIPTION # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

	TBD
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# AUTHOR(S) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	Original Authors (matlab and C code): Haim Avron and Sivan Toledo.
	
	See: Haim Avron, Petar Maymounkov, and Sivan Toledo. "Blendenpik: Supercharging LAPACK's least-squares solver." 
		 SIAM Journal on Scientific Computing, 32 (3), 2010

	Port to this python code (and further edits): W. Ross Morrow <morrowwr@gmail.com>
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# NOTES # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	2/9/2017 - Code created by W. Ross Morrow (WRM)
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# TESTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	2/10/2017 - Hadamard transform routines, "rht" and sub-routines, tested and passed. See codes below. 
	2/10/2017 - Preconditioned LSQR code, "bnp_lsqr", tested and passed. See codes below. Note that solution may be off, 
				but the absolute/relative residuals at a solution is as nearly good as possible. 
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# TODO  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	1. finish FRUT calls... DCT, DHT implementation, FFTW "wisdom"
	2. underdetermined systems?
	3. change time to optional, and use timeit where possible
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# COPYRIGHT # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	Copyright and License for this python code - Apache 2.0+ license
	==========================================================================
	Copyright (c) 2017, W. Ross Morrow. All rights reserved.
	
	THIS SOFTWARE PROVIDED "AS-IS" WITH NO WARRANTY. 
	
	
	Copyright and License from original matlab/C code - BSD license
	==========================================================================
	Copyright (c) 2009-2016, Haim Avron and Sivan Toledo. All rights reserved.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions are met:
	
		* Redistributions of source code must retain the above copyright
		  notice, this list of conditions and the following disclaimer.
		* Redistributions in binary form must reproduce the above copyright
		  notice, this list of conditions and the following disclaimer in the
		  documentation and/or other materials provided with the distribution.
		* Neither the name of Tel-Aviv University nor the
		  names of its contributors may be used to endorse or promote products
		  derived from this software without specific prior written permission.

	THIS SOFTWARE IS PROVIDED BY HAIM AVRON AND SIVAN TOLEDO ''AS IS'' AND ANY
	EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	DISCLAIMED. IN NO EVENT SHALL <copyright holder> BE LIABLE FOR ANY
	DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from time import time # take out in favor of timeit at some point
import numpy as np
from numpy.random import rand , randn
from numpy.linalg import lstsq , cond , norm
from scipy.linalg import qr , solve_triangular , hadamard # use scipy's? or numpy's? 
from scipy.sparse.linalg import LinearOperator , aslinearoperator , lsqr

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def rht_r( A , K , L ) :

	"""
		Actual recursion. We focus in a bit on the basic operation here. We recurse computing products like
		
			U = HP(A|K,L) = H A[K:L,:]
		
		where K < L, K,L in {0} \/ {2^k : k = 1,2,3,...}. For example, 
			
			U_m = HP( A | 0 , 2^(m-1) ) and V_m = HP( A | 2^(m-1) , 2^m ) where m = ceil( log2( M ) ), M = A.shape[0]
		
		A simple example, our termination condition, is when L = K+2: 
		
			U = HP(A|K,K+2) = H A[K:K+2,:]
			  = 1/sqrt(2) [ 1 ,  1 ] [ A[ K ,:] ]
						  [ 1 , -1 ] [ A[K+1,:] ]
			  = 1/sqrt(2) [ 1 ] A[K,:] + [  1 ] A[K+1,:]
						  [ 1 ]        + [ -1 ] 
			  = 1/sqrt(2) [ A[K,:] + A[K+1,:] ]
						  [ A[K,:] - A[K+1,:] ]
		
		Otherwise, we have
		
			   HP( A |  0  , 2^l ) , l > 1 (otherwise L = 2^1 = 2 = 2+0 = K+2)
			or HP( A | 2^k , 2^l ) , k > 1 or l > 2 if k == 1 (otherwise K = 2, L = 2^2 = 4, and L = K+2)
		
		We want to split in half, which is easiest when K == 0: 
			
			A[ 0 : 2^l , : ] -> [ A[   0     : 2^(l-1) ] ]
								[ A[ 2^(l-1) :   2^l   ] ]
								
		Note that 2^(l-1) = 2^l/2 = L/2. When K > 0, 
			
			A[2^k:2^l,:] -> A[ 2^k + 0:2^l-2^k ,:] -> [ A[   2^k + 0:2^(l-1)-2^(k-1)   , : ] ]
													  [ A[ 2^k + 2^(l-1)-2^(k-1) : 2^l , : ] ]
													  
		Here note that 2^(l-1) = L/2 and 2^(k-1) = K/2, so that (of course) 2^(l-1)-2^(k-1) = (L-K)/2. 
			
	"""
	if L < K : return rht_r( A , L , K ) # a bit of argument check
	if L - K == 2 : # recursion termination condition
		Kp1 = K + np.uint64(1)
		if A.shape[0] < Kp1 : return 0.70710678118654746 * np.vstack( [ A[K,:] , A[K,:] ] ) 
		else : return 0.70710678118654746 * np.vstack( [ A[K,:] + A[Kp1,:] , A[K,:] - A[Kp1,:] ] ) 
	else : 
		S = np.uint64( L/2 )
		if K > 0 : S = np.uint64( K/2 ) + S
		if A.shape[0] <= S : # second split is all zeros, lying in the implicitly padded portion of A
			U = rht_r( A , K , S )
			return 0.70710678118654746 * np.vstack( [ U , U ] ) 
		else : # second half split overlaps filled portion of A
			U , V = rht_r( A , K , S ) , rht_r( A , S , L )
			return 0.70710678118654746 * np.vstack( [ U + V , U - V ] ) 

def rhdt_r( A , D , K , L ) :
	""" modified version of the call above, accounting for diagonal "weights" D
		
			U = HDP(A|D,K,K+2) = HP(DA|K,K+2)
			  = 1/sqrt(2) [ 1 ,  1 ] [ (DA)[ K ,:] ]
						  [ 1 , -1 ] [ (DA)[K+1,:] ]
			  = 1/sqrt(2) [ 1 ] D[K] A[K,:] + [  1 ] D[K+1] A[K+1,:]
						  [ 1 ]             + [ -1 ] 
			  = 1/sqrt(2) [ D[K] A[K,:] + D[K+1] A[K+1,:] ]
						  [ D[K] A[K,:] - D[K+1] A[K+1,:] ]
		
		
	"""
	if L < K : return rhdt_r( A , D , L , K ) # a bit of argument check
	if L - K == 2 : # recursion termination condition
		Kp1 = K + np.uint64(1)
		U = D[K] * A[K,:]
		if A.shape[0] < Kp1 : return 0.70710678118654746 * np.vstack( [ U , U ] ) 
		else : 
			V = D[Kp1] * A[Kp1,:]
			return 0.70710678118654746 * np.vstack( [ U + V , U - V ] ) 
	else : 
		S = np.uint64( L/2 )
		if K > 0 : S = np.uint64( K/2 ) + S
		if A.shape[0] <= S : # second split is all zeros, lying in the implicitly padded portion of A
			U = rhdt_r( A , D , K , S )
			return 0.70710678118654746 * np.vstack( [ U , U ] ) 
		else : # second half split overlaps filled portion of A
			U , V = rhdt_r( A , D , K , S ) , rhdt_r( A , D , S , L )
			return 0.70710678118654746 * np.vstack( [ U + V , U - V ] ) 

def rht( A , D=None ) : 
	
	""" Recursive Hadamard Transform. These matrices are constructed as
		
			H_1     = 1/sqrt(2) [  1  ,  1  ;  1  , - 1  ]
			H_{k+1} = 1/sqrt(2) [ H_k , H_k ; H_k , -H_k ] for k >= 1
		
		So a M x N matrix A can be transformed by H_m where m = ceil( log2(M) ) applied to Ap = [ A ; 0 ] as follows: 
			
			H_m Ap  = 1/sqrt(2) [ H_{m-1} ,  H_{m-1} ] [ Ap[    0    : 2^(m-1) , : ] ]
								[ H_{m-1} , -H_{m-1} ] [ Ap[ 2^(m-1) :   2^m   , : ] ]
				    = 1/sqrt(2) [ H_{m-1} ] Ap[0:2^(m-1),:] + [  H_{m-1} ] Ap[2^(m-1):2m,:]
								[ H_{m-1} ]					  [ -H_{m-1} ] Ap[2^(m-1):2m,:]
				    = 1/sqrt(2) [ U_m + V_m ]
								[ U_m - V_m ]
		
		where U_m = H_{m-1} Ap[0:2^(m-1) ,:] and V_m = H_{m-1} Ap[2^(m-1):2m,:]. This sets up a recursion we can use. 
		
	"""
	m = np.uint64( np.ceil( np.log2( A.shape[0] ) ) )
	if D is None : return rht_r( A , np.uint64(0) , np.uint64(2**m) )
	else : return rhdt_r( A , D , np.uint64(0) , np.uint64(2**m) )
	
def test_hadamard(  ) : 
	
	print( 'testing hadamard routine...' )
	for n in range(1,11) : 
		N = 2 ** n
		H = 1.0 / np.sqrt(N) * hadamard( N ) # normalized hadamard matrix, unitary
		I = np.eye(N)
		Hr = rht( I ) # this should construct the N x N normalized hadamard matrix
		print( '  n = %d ( N = %d ): %0.16f' % ( n , N , np.max( np.max( H - Hr ) ) ) )
	print( '' )
	
def test_hadamard_D(  ) : 
	
	print( 'testing diagonal-then-hadamard routine...' )
	for n in range(1,11) : 
		N = 2 ** n
		D = rand( N )
		H = 1.0 / np.sqrt(N) * hadamard( N ) # normalized hadamard matrix, unitary
		Hd = H * D # effectively H * diag(D)
		I = np.eye(N)
		Hr = rht( I , D=D ) # should construct H diag(D)
		print( '  n = %d ( N = %d ): %0.16f' % ( n , N , np.max( np.max( Hd - Hr ) ) ) )
	print( '' )

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def frut( A , D , type='DHT' ) : 
	
	# THIS will take some work... the transforms appear complicated, at least as implemented by Avron etc. 
	# currently implemented with a naive, recursive implementation of a Hadamard transform, but this blows up the 
	# size of the problem because we transform the M x N matrix A into the 2^(ceil(log2(M))) x N matrix HA. 
	# 
	# Presuming HA is dense (as I think we must), could this significantly impact problem size? Well, the 
	# "worst" we could do is to have M = 2^(m-1) + 1, so that L = 2^m and (L-M)/M = 1 - 1/2^(m-1). As M/m 
	# grows, this approaches one (and quickly) so that the amount of memory required can, basically, 
	# DOUBLE with a full Hadamard transform. That's not good, but not awful either... especially for dense
	# solve size matrices. 
	
	return rht( A , D ) # TEMPORARY, I WOULD ASSUME

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def rsp( A , params , b=None ) : 

	""" random sample preconditioner """

	if params['improve_start_point'] and ( b is None ) : 
		raise ValueError( 'rsp needs b to improve starting point' )
	
	m , n = A.shape
	mm = m
	times = 0
	timing = {}
	
	B = A.copy()

	timing['qr_time'] , timing['condest_time'] , timing['sample_time'] , timing['frut_time'] = 0 , 0 , 0 , 0

	if not params['improve_start_point'] : x0 = None

	htimes = params['preproc_steps']
	while True : 
		
		# fast random unitary transformation (FRUT) step
		t0 = time()
		for i in range(0,htimes) : # 1:htimes
			D = np.sign( randn( mm , 1 ) ) # random Rademacher variables
			B = frut( B , D , params['mix_type'] ) # is this "overwriting" of A ok? Does this overwrite? 
			mm = B.shape[0] # a bit repetitive? depends on the transform
			if params['improve_start_point'] : b = frut( b , D , params['mix_type'] )
		frut_time = time() - t0
		if params['verbose'] : print( '\t\tRandom unit diagonal + unitary transformation time: %0.6f sec' % frut_time )
		timing['frut_time'] += frut_time
		
		# sampling step
		t0 = time()
		t = min( params['gamma'] * n / mm , 1 )
		rows = ( rand( mm ) < t )
		R = B[ rows , : ] # initialize R as a (random) row-selection of A
		sample_time = time() - t0
		if params['verbose'] : print( '\t\tRandom sampling time: %0.6f sec' % sample_time )
		timing['sample_time'] += sample_time
		
		# Suppose we are selecting r rows. Then here we are basically finishing something like
		# 
		#    R = [ I ] P (H_T D_T) ... (H_1 D_1) [ A ] 
		#        [ 0 ] 							 [ 0 ] (if padding A for up-mixing transforms)
		# 
		# where I is an r x r identity matrix, P is a L x L permutation matrix, H_t is a
		# L x L unitary transformation, D is an L x L diagonal Rademacher variable matrix, 
		# and T is the number of transformations; L >= M for M x N A. Another way to look
		# at this is 
		# 
		#    R = Z' [ A ] 
		#        	[ 0 ] (if padding A for up-mixing transforms)
		# 
		# where the L x r matrix Z is defined by 
		# 
		#    Z = (H_1 D_1)' ... (H_T D_T)' P' [ I , 0' ]
		#      = (D_1 H_1') ... (D_T H_T') P' [ I , 0' ]
		# 
		# because D_t is symmetric. Hadamard transforms are symmetric (induction proof), which means 
		# 
		#    Z = (D_1 H_1) ... (D_T H_T) Q    where   Q = P' [ I , 0' ]
		# 
		# This is a better approach, I think, because we can form Q, compute 
		# 
		#    Z <- Q
		#    for t = 1:T, 
		# 	 	Z <- H_t Z
		# 	 	Z <- D_t Z
		# 
		# then take R <- Z' A. 
		# 
		# Is this useful? We sample a row with probability p = g N / L, so r, the number of rows sampled, 
		# has a Binomial distribution with parameter p. This means Expect( r ) = L p = g N, and hence
		# Z should be expected to be LARGER than a transformed A matrix when g > 1. Because the Binomial
		# is a roughly symmetric distribution (median ~ mean), we should also have Z larger more than
		# half of the time. We also would have to transform and THEN compute a matrix-matrix product, 
		# which we don't have to do in the process above. That is, DON'T do things this way. 
		
		# modify R matrix for "slight coherence" (?) condition
		if params['slight_coherence'] > 0 : 
			R = np.hstack( [ R , np.max( np.max(R) ) * rand( params['slight_coherence'] , n ) ] )
		
		# QR factorization step
		t0 = time()
		S = qr( R , mode='r' )[0]
		S = S[0:S.shape[1],0:S.shape[1]]
		if params['improve_start_point'] : 
			x0 = R.T.dot( b[rows] ) # x0 <- R' b[rows]
			solve_triangular( S , x0 , trans='T' , overwrite_b=True ) # x0 <- inv(S') x0
			solve_triangular( S , x0 , trans='N' , overwrite_b=True ) # x0 <- inv(S ) x0
		qr_time = time() - t0
		if params['verbose'] : print( '\t\tQR on random sample time: %0.6f sec' % qr_time )
		timing['qr_time'] += qr_time
		
		# estimate condition number (easier for a triangular matrix?) step
		t0 = time()
		ce = cond( R ) # dtrcon is reciprocal condition number estimation, numpy.linalg.cond is condition number est
		condest_time = time() - t0
		if params['verbose'] : print( '\t\tCondition estimation: %0.6f sec' % condest_time )
		timing['condest_time'] += condest_time
		
		# increment times
		times += 1
		
		# check if we're done
		if ce < params['maxcond'] : return S , True , timing , x0
		else : 
			if times <= 3 : 
				if params['verbose'] : print( '\t\tFailed to produced a non singular preocnditioner... applying one more FRUT...' )
				htimes = 1
			else : 
				if params['verbose'] : print( '\t\tFailed to produced a non singular preconditioner... too many FRUTs... giving up...' )
				return None , False , timing , x0
				
	# NOTE no return here. Pretty sure that's ok... 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def bnp_lsqr( A , b , R , tol=1.0e-8 , max_iter=None ) : 
	""" specialized LSQR routine, basically meant to wrap preconditioning """
	# Ao = aslinearoperator( A )
	m , n = A.shape
	def plsqr__matvec( v ) : 
		solve_triangular( R , v , overwrite_b=True )
		return A.dot( v )
	def plsqr_rmatvec( v ) : 
		v = A.T.dot( v )
		solve_triangular( R , v , overwrite_b=True )
		return v
	AinvR = LinearOperator( shape=(m,n) , matvec=plsqr__matvec , rmatvec=plsqr_rmatvec )
	x = lsqr( AinvR , b , damp=0.0 , atol=tol , iter_lim=1000 )[0]
	solve_triangular( R , x , overwrite_b=True )
	x = x.reshape( (x.size,1) ) # needed? I think so. 
	return x
	
def test_bnp_lsqr( M , N ) :

	print( 'testing preconditioned LSQR method...' )
	
	A , b = randn( M , N ) , randn( M , 1 )
	
	xT = lstsq( A , b )[0]
	
	# If A = Q[R;0], then preconditioning with R has us solve A inv(R) y = b for y, obtaining x from x = inv(R) y. 
	# But A inv(R) = Q[I;0] should be very well-conditioned. Here let us solve with LSQR explitly forming A inv(R) 
	# via Q matrix construction (using economic mode)
	Q , R = qr( A , mode='economic' )
	xt = lsqr( aslinearoperator( Q ) , b )[0] #, damp=0.0 , atol=tol , iter_lim=max_iter )[0]
	solve_triangular( R , xt , overwrite_b=True ) # finish 
	
	xt = xt.reshape( (xt.size,1) )
	
	print( '  sanity check: ' )
	print( '    absolute (relative) error in solutions: %0.16f (%0.16f)' % ( np.max( np.abs( xT - xt ) ) , np.max( np.abs( xT - xt ) / np.abs( xT ) ) ) )
	r = b - A.dot(xt)
	print( '    absolute (relative) residual for solns: %0.16f (%0.16f)' % ( np.max( np.abs( r ) ) , np.max( np.abs( r ) / np.abs( b ) ) ) )
	
	xb = bnp_lsqr( A , b , R )
	
	print( '  method used: ' )
	print( '    absolute (relative) error in solutions: %0.16f (%0.16f)' % ( np.max( np.abs( xT - xb ) ) , np.max( np.abs( xT - xb ) / np.abs( xT ) ) ) )
	r = b - A.dot(xb)
	print( '    absolute (relative) residual for solns: %0.16f (%0.16f)' % ( np.max( np.abs( r ) ) , np.max( np.abs( r ) / np.abs( b ) ) ) )
	
	print( '' )

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def blendenpik_under( A , b , params ) : 
	
	tstart = time()
	
	timing = {}

	# build preconditioner
	t1 = time()
	R , flag , timing['precond_timing'] = rsp( A.T , params )
	L = R.T
	timing['precond_total_time'] = time() - t1
	if params['verbose'] : print( '\tBuilding preconditioner time: %0.6f sec' % timing.precond_total_time )

	# Solve
	if flag : 
		t1 = time()
		x , timing['lsqr_its'] = bnp_lsqr( A , b , R , params['tol'] , params['max_iter'] )
		timing['lsqr_time'] = time() - t1
		if params['verbose'] : print( '\tLSQR time: %0.6f sec' % timing['lsqr_time'] )
	else : 
		if params['verbose'] : print( 'Failed to get a full rank preconditioner. Using standard OLS.' )
		t1 = time()
		try : x = lstsq( A , b )[0]
		except Exception as e : 
			x = lsqr( aslinearoperator(A) , b , atol=params['tol'] , iter_lim=params['max_iter'] )
		timing['standard_time'] = time() - t1
		if params['verbose'] : print( '\tStandard time: %0.6f sec' % timing['standard_time'] )
	
	timing['total_time'] = time() - tstart
	if params['verbose'] : print( 'Total time: %0.6f sec' % timing['total_time'] ) 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def blendenpik_over( A , b , params ) : 

	tstart = time()
	
	timing = {}
	
	# Build preconditioner
	if params['verbose'] : print( '\tBuilding preconditioner...' )
	t1 = time()
	R , flag , timing['precond_timing'] , x0 = rsp( A , params , b )
	timing['precond_total_time'] = time() - t1
	if params['verbose'] : print( '\tBuilding preconditioner time: %0.6f sec' % timing['precond_total_time'] )
	
	# Solve
	if flag : # preconditioning was ok
		
		t1 = time()
		if not params['improve_start_point'] :
			x = bnp_lsqr( A , b , R , params['tol'], params['max_iter'] )
		else : 
			r0 = b - A.dot( x0 )
			dx = bnp_lsqr(A, r0, R, params['tol'] * norm(b) / norm(r0), params['max_iter'] )
			x = x0 + dx
		timing['lsqr_time'] = time() - t1
		if params['verbose'] : print( '\tLSQR time: %0.6f sec' % timing['lsqr_time'] )
		
	else : # preconditioning failed, revert to standard method
		
		if params['verbose'] : print( 'Failed to get a full rank preconditioner. Using standard OLS (QR/LSQR).' )
		t1 = time()
		try : x = lstsq( A , b )[0]
		except Exception as e : 
			x = lsqr( aslinearoperator(A) , b , atol=params['tol'] , iter_lim=params['max_iter'] )
		timing['standard_time'] = time() - t1
		if params['verbose'] : print( '\tStandard time: %0.6f sec' % timing['standard_time'] )
		
	timing['total_time'] = time() - tstart
	if params['verbose'] : print( 'Total time: %0.6f sec' % timing['total_time'] )
	
	return x , timing

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def blendenpik( A , b , 
				mix_type='DHT' , 
				gamma=4.0 , 
				preproc_steps=1 , 
				maxcond=None , 
				tol=1.0e-14 , 
				max_iter=1000 , 
				verbose=True ) : 
	"""
		Solve the LS problem min || A x - b ||_2 using Blendenpik. 
		 
		parameters are as follows:
		
			mixtype - type of mixing transform. Optional values: 'DCT', 'DHT', 'WHT'. 
			gamma - gamma * min(n,m) rows/columns will be sampled (A is m-by-n). 
			preproc_steps - number of mixing steps to do in advance. 
			maxcond - maximum condition number of the preconditioner.
			tol - convergence thershold for LSQR.
			maxit - maximum number of LSQR iterations.
			lsvec - whether to output in "timing" the LSQR residuals.

		Output:
			
		    x - the solution.
		    timing - statistics on the time spent on various phases.
				  
		From matlab code 
			6-December 2009, Version 1.3
			Copyright (C) 2009, Haim Avron and Sivan Toledo.
			
		Adapted to python
			Started 9 February 2017
			W. Ross Morrow (morrowwr@gmail.com)
			

	"""
	
	m , n = A.shape
	
	if maxcond is None : maxcond = 1.0 / ( 5.0 * np.finfo(float).eps )
	
	params = { 'mix_type' : mix_type , 
				'gamma' : gamma ,  
				'preproc_steps' : preproc_steps , 
				'maxcond' : maxcond , 
				'tol' : tol , 
				'max_iter' : max_iter , 
				'slight_coherence' : 0 , 
				'improve_start_point' : True , 
				'verbose' : verbose }
	
	x , timing = blendenpik_over(A,b,params) if m >= n else blendenpik_under(A,b,params) 
	
	return x

def test_blendenpik( M , N ) : 
	
	print( 'testing blendenpik itself...' )
	A , b = randn( M , N ) , randn( M , 1 )
	t = time()
	xb = blendenpik( A , b )
	t = time() - t
	r = b - A.dot( xb )
	print( '  blendinpick residual norm: %0.16f (%0.16f) (%0.6f sec)' % ( np.max( np.abs( r ) ) , np.max( np.abs( r ) / np.abs( b ) ) , t ) )
	
	t = time()
	xt = lstsq( A , b )[0]
	t = time() - t
	r = b - A.dot( xt )
	print( '     np.lstsq residual norm: %0.16f (%0.16f) (%0.6f sec)' % ( np.max( np.abs( r ) ) , np.max( np.abs( r ) / np.abs( b ) ) , t ) )
	
	print( '' )

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def test_all(  ) : 
	
	test_hadamard()
	
	test_hadamard_D()
	
	M , N = 1000 , 10
	test_blendenpik( M , N )

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == '__main__' : 
	
	print( "\nblendnpik.py\n" )
	
	# test_hadamard() # PASSED
	# test_hadamard_D() # PASSED
	# test_bnp_lsqr( 1000 , 100 ) # PASSED
	test_blendenpik( 100 , 10 )
	