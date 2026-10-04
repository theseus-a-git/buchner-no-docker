% Author: Ben Carlson
% Description: This script generates random values in the same order as 
% crcbpso.m and saves them in a .txt file. These values can be read into
% the Python version as premade_Rs for cross-validation.

% ----------------------------
% Parameters
popsize = 5;
nDim = 4;
nIterations = 2000;
max_velocity = 0.5;
filename1 = 'data_pos_init.txt';
filename2 = 'data_vel_init.txt';
filename3 = 'data_upd.txt';
% ----------------------------

% Fixed RNG seed for reproducibility
rng(1);

% Generate the random values to initialize the positions and velocities
positions = rand(popsize, nDim);
velocities = rand(popsize, nDim);

% Store initialization values in row-major order.
%
% MATLAB stores matrices in column-major order, so transpose the matrix
% before flattening it. This produces the ordering:
%
%   positions(1,1), positions(1,2), ..., positions(1,nDim),
%   positions(2,1), positions(2,2), ..., positions(2,nDim),
%   ...
%
% which matches NumPy's default reshape(order='C').
% Store the values from initialization
pos_vals = reshape(positions.', 1, []); 
vel_vals = reshape(velocities.', 1, []); 

% Store the position initialization values 
fid1 = fopen(filename1, 'w'); 
fprintf(fid1, '%.25g\n', pos_vals); 
fclose(fid1); 
% Store the velocity initialization values 
fid2 = fopen(filename2, 'w'); 
fprintf(fid2, '%.25g\n', vel_vals); 
fclose(fid2);

% Generate random vals used in R1 and R2
fid3 = fopen(filename3, 'w');
for iter = 1:1:nIterations
    R1_matrix = zeros(1, popsize*nDim);
    R2_matrix = zeros(1, popsize*nDim);
    for p = 1:1:popsize
        R1 = rand(1, nDim);
        R2 = rand(1, nDim);
        idx = (p-1)*nDim + (1:nDim);
        R1_matrix(idx) = R1;
        R2_matrix(idx) = R2;
    end
    % Add the random value matrices
    fprintf(fid3, '%.25g\n', R1_matrix);
    fprintf(fid3, '%.25g\n', R2_matrix);
end

fclose(fid3);

disp("data files created successfully")
