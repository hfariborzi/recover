import streamlit as st
import numpy as np
import pandas as pd
import time

def process_betas(betas_str):
    # Remove brackets if any
    betas_str = betas_str.strip("[]()")
    # Split based on comma or semicolon and convert to floats
    betas = list(map(float, betas_str.replace(';', ',').split(',')))
    return betas

def completely_random_matrices_ui(betas, reps, sample_size, threshold_cutoff_base):
    betas = process_betas(betas)  # Convert the string input to list of floats
    start_time = time.perf_counter()
    # Call your function here
    df, duration = completely_random_matrices(betas, reps, sample_size, threshold_cutoff_base)  # Ensure df is returned here
    end_time = time.perf_counter()
    duration = end_time - start_time
    return df, f"The calculations were done in {duration:.3f} seconds."  # Make sure df is included in return


def generate_positive_definite_correlation_matrix(n):
    A = np.random.randn(n, n)
    A = (A + A.T) / 2
    A += n * np.eye(n)  # Make it definitely positive definite
    _, eigenvectors = np.linalg.eigh(A)
    A = eigenvectors @ np.diag(np.abs(np.random.rand(n))) @ eigenvectors.T
    return A

def ensure_positive_definite(R):
    """ Adjust the matrix to ensure it is positive definite """
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    min_eigenvalue = eigenvalues.min()
    if min_eigenvalue <= 0:
        # Adjust eigenvalues to be positive
        eigenvalues[eigenvalues <= 0] = 0.01
    R = np.dot(eigenvectors, np.dot(np.diag(eigenvalues), eigenvectors.T))
    np.fill_diagonal(R, 1)  # Ensure diagonal is 1
    return R

def generate_correlation_matrix(d):
    max_attempts = 1000
    for attempt in range(max_attempts):
        R = np.eye(d)
        # Fill the first off-diagonal
        for i in range(d-1):
            R[i, i+1] = R[i+1, i] = np.random.uniform(-1, 1)

        # Fill remaining off-diagonals
        for k in range(2, d):
            for i in range(d-k):
                j = i + k
                partial = np.random.uniform(-1, 1)
                potential_correlation = partial * np.sqrt((1 - R[i, j-1]**2) * (1 - R[i+1, j]**2))
                if -1 <= potential_correlation <= 1:
                    R[i, j] = R[j, i] = potential_correlation
        
        if np.all(np.linalg.eigvals(R) > 0):
            return R  # Return if positive definite

    # Ensure positive definiteness if failed in attempts
    return ensure_positive_definite(R)

def lower_tri_index_to_var_pair(index, var_size):
    counter = 0
    for i in range(1, var_size):
        for j in range(i + 1, var_size + 1):
            counter += 1
            if counter == index:
                return (i, j)
    return None

def completely_random_matrices(betas, reps, samplesize, threshold_cutoff_base=0.001):
    n = len(betas)
    var_size = n + 1
    k = len(betas) + (n * (n - 1)) // 2
    output_df = pd.DataFrame()
    successful_reps = 0
    number_cases = 0

    while successful_reps < reps:
        random_cor_matrix = generate_correlation_matrix(n)
        corr_matrix = np.dot(np.array(betas), random_cor_matrix).flatten()
        new_matrix1 = np.vstack((corr_matrix, random_cor_matrix))
        first_column = np.vstack([np.array([[1]]), corr_matrix.reshape(-1, 1)])
        new_matrix2 = np.hstack([first_column, new_matrix1])
        rounded_matrix = np.round(new_matrix2, 3)

        if np.all(np.linalg.eigvals(rounded_matrix) > 0):
            estimates = np.linalg.inv(rounded_matrix[1:, 1:]).dot(rounded_matrix[0, 1:])
            threshold = np.mean(np.abs(betas - estimates))
            number_cases += 1

            if threshold < threshold_cutoff_base + round(successful_reps / 1000000, 3):
                successful_reps += 1

                bottom_cor_Est = random_cor_matrix[np.tril_indices(n, -1)]
                new_row = np.hstack([corr_matrix, bottom_cor_Est])
                output_df = pd.concat([output_df, pd.DataFrame([new_row])], ignore_index=True)

    print(f"Repetitions required for estimation is {number_cases}")

    # Additional outputs for variable pairs
    cor_output = np.zeros(k)
    n_output = np.zeros(k)
    if successful_reps > 0:
        for i in range(1, k + 1):
            print(i)
            index = i
            vars = lower_tri_index_to_var_pair(index, var_size)
            if vars:
                variable_pair = f"Variable {vars[0]} and Variable {vars[1]}"
                print(variable_pair)
                column_data = output_df.iloc[:, i-1].values
                column_avg = np.mean(column_data)
                column_sd = np.std(column_data)
                try:
                    column_fisher = np.arctanh(np.clip(column_data, -0.999999, 0.999999))
                    column_avg_fisher = np.tanh(np.mean(column_fisher))
                    column_sd_fisher = np.std(column_fisher)
                    column_var_fisher = column_sd_fisher**2
                    totalvar = column_var_fisher + (1/(samplesize-3)**0.5)**2
                    ExpectedN = (3 * totalvar + 1) / totalvar
                    ExpectedN = round(ExpectedN, 1)
                    n_output[i-1] = ExpectedN
                    cor_output[i-1] = column_avg
                    print(f"Expected correlation {i} based on Fisher is {column_avg_fisher}")
                    print(f"Expected (Pseudo) N is {ExpectedN}")
                    #titletext = f"Histogram of Correlation {variable_pair}"
                    #plt.hist(column_data, bins=np.linspace(-1, 1, 21), color='gray', edgecolor='black')
                    #plt.title(titletext)
                    #plt.xlabel('Correlations')
                    #plt.ylabel('Frequency')
                    #plt.show()
                except Exception as e:
                    print("Error processing Fisher Z-transform:", e)

        # Matrix creation and assignment
        mat1 = np.full((var_size, var_size), np.nan)
        np.fill_diagonal(mat1, np.nan)
        tri_indices = np.tril_indices(var_size, -1)
        mat1[tri_indices] = np.round(cor_output[:len(tri_indices[0])], 3)
        mat1[tri_indices[1], tri_indices[0]] = np.round(n_output[:len(tri_indices[0])], 3)  # Transposed indices for upper triangle

        var_names = [f"Var{i+1}" for i in range(var_size)]
        df = pd.DataFrame(mat1, index=var_names, columns=var_names)

        return df, number_cases  # Return the final DataFrame and any other info you want to display

    # If no successful reps, return an empty DataFrame with a message or similar
    return pd.DataFrame(), 0


st.title("Correlation Matrix Generator-With Joe's Generator (simulated)")

betas_input = st.text_input("Betas (comma or semicolon separated)", "-0.2, 0.4")
reps_input = st.number_input("Number of Repetitions", min_value=1, value=500)
sample_size_input = st.number_input("Sample Size", min_value=1, value=200)
threshold_input = st.number_input("Threshold Cut-off", min_value=0.0001, max_value=1.0, value=0.001)

if st.button("Run"):
    df, duration_message = completely_random_matrices_ui(betas_input, reps_input, sample_size_input, threshold_input)
    st.write(duration_message)
    st.dataframe(df)
