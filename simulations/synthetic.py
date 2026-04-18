# synthetic asym function
def synth_asym_func1_ver1(X, Z, task_type):

    # Extract individual features from X and Z matrices
    X1, X2, X3, X4, X5  = X[:,0], X[:,1], X[:,2], X[:,3], X[:,4]
    X6, X7, X8, X9, X10 = X[:,5], X[:,6], X[:,7], X[:,8], X[:,9]
    X11, X12, X13, X14, X15 = X[:,10], X[:,11], X[:,12], X[:,13], X[:,14]
    Z1, Z2, Z3, Z4, Z5 = Z[:,0], Z[:,1], Z[:,2], Z[:,3], Z[:,4]
    Z6, Z7, Z8, Z9, Z10 = Z[:,5], Z[:,6], Z[:,7], Z[:,8], Z[:,9]
    Z11, Z12, Z13, Z14, Z15 = Z[:,10], Z[:,11], Z[:,12], Z[:,13], Z[:,14]

    # Define two-view interactions between X and Z
    interaction1 = + X1 * Z1
    interaction2 = - X2 * Z1
    interaction3 = + X3 * Z1
    interaction4 = - X4 * Z2
    interaction5 = + X4 * Z3
    interaction6 = - X4 * Z4
    interaction7 = + X5 * Z5
    interaction8 = - X6 * Z5
    interaction9 = + X7 * Z6
    interaction10 = - X8 * Z7
    interaction11 = + X8 * Z8
    interaction12 = - X9 * Z9
    interaction13 = + X9 * Z10
    interaction14 = - X10 * Z11
    interaction15 = + X11 * Z11
    interaction16 = - X12 * Z12
    interaction17 = + X13 * Z12

    # Define within-view effects for X and Z
    X_within_effects = X1 - 2*X2 + X3 - X4 * X5 + X6 - X7 + X8 - X9 + X10 * X11 - X12 + X13 - X14 + X15 - X10 * X13 + X14 * X15
    Z_within_effects = Z1 - Z2 + Z3 - Z4 + 2*Z5 - Z6 * Z7 + Z8 - Z9 + Z10 - Z11 + Z12 - Z13 + Z14 - Z15 + Z13 * Z14 - Z3 * Z15 + Z14 * Z15

    # Genereate the linear output
    linear_output = (
        interaction1 + interaction2 + interaction3 + interaction4 + interaction5 +
        interaction6 + interaction7 + interaction8 + interaction9 + interaction10 +
        interaction11 + interaction12 + interaction13 + interaction14 + interaction15 +
        interaction16 + interaction17 + X_within_effects + Z_within_effects
    )

    # Define the ground truth for within-view interactions
    ground_truth = [ (1,1), (2,1), (3,1), (4,2), (4,3), (4,4), (5,5), (6,5), (7,6), (8,7), (8,8), (9,9), (9,10), (10,11), (11,11), (12,12), (13,12)]

    # Generate the response variable Y based on the task type (regression or classification)
    if task_type == "regression":
        Y = linear_output
    elif task_type == "classification":
        Y = generate_binary_response(linear_output)

    return Y, ground_truth


def synth_asym_func2_ver1(X, Z, task_type):

    # Extract individual features from X and Z matrices
    X1, X2, X3, X4, X5  = X[:,0], X[:,1], X[:,2], X[:,3], X[:,4]
    X6, X7, X8, X9, X10 = X[:,5], X[:,6], X[:,7], X[:,8], X[:,9]
    X11, X12, X13, X14, X15 = X[:,10], X[:,11], X[:,12], X[:,13], X[:,14]
    Z1, Z2, Z3, Z4, Z5 = Z[:,0], Z[:,1], Z[:,2], Z[:,3], Z[:,4]
    Z6, Z7, Z8, Z9, Z10 = Z[:,5], Z[:,6], Z[:,7], Z[:,8], Z[:,9]
    Z11, Z12, Z13, Z14, Z15 = Z[:,10], Z[:,11], Z[:,12], Z[:,13], Z[:,14]

    # Define two-view interactions between X and Z
    interaction1 = + X1 * Z1
    interaction2 = - np.log(np.abs(X2 + Z1))
    interaction3 = + np.sin(X3 * np.sin(Z1))
    interaction4 = - np.abs(X4 * Z2)
    interaction5 = + X4 * Z3
    interaction6 = - np.log(np.abs(X4 + Z4))
    interaction7 = + np.sin(X5 * np.sin(Z5))
    interaction8 = - np.abs(X6 * Z5)
    interaction9 = + X7 * Z6
    interaction10 = - np.log(np.abs(X8 + Z7))
    interaction11 = + np.sin(X8 * np.sin(Z8))
    interaction12 = - np.abs(X9 * Z9)
    interaction13 = + X9 * Z10
    interaction14 = - np.log(np.abs(X10 + Z11))
    interaction15 = + np.sin(X11 * np.sin(Z11))
    interaction16 = - np.abs(X12 * Z12)
    interaction17 = + X13 * Z12

    # Define within-view effects for X and Z
    X_within_effects = X1 - np.sin(2 * X2) + X3**2 - X4 + X5 * X11 - np.sqrt(np.abs(2 * X6)) + X7 - X8 + np.abs(2 * X9) - X10 * X15 + X12 - np.sqrt(np.abs(2 * X13)) + X14**2
    Z_within_effects = Z1 - 1/(1 + Z2**2) + Z3*Z15 - Z4**2 + Z5 - Z6 + np.sin(Z7) - Z8 + Z9**2 - Z10 + Z11**2 - Z12 +  np.log(np.abs(Z13 + Z14))

    # Genereate the linear output
    linear_output = (
        interaction1 + interaction2 + interaction3 + interaction4 + interaction5 +
        interaction6 + interaction7 + interaction8 + interaction9 + interaction10 +
        interaction11 + interaction12 + interaction13 + interaction14 + interaction15 +
        interaction16 + interaction17 + X_within_effects + Z_within_effects
    )

    # Define the ground truth for within-view interactions
    ground_truth = [ (1,1), (2,1), (3,1), (4,2), (4,3), (4,4), (5,5), (6,5), (7,6), (8,7), (8,8), (9,9), (9,10), (10,11), (11,11), (12,12), (13,12)]

    # Generate the response variable Y based on the task type (regression or classification)
    if task_type == "regression":
        Y = linear_output
    elif task_type == "classification":
        Y = generate_binary_response(linear_output)

    return Y, ground_truth


def synth_asym_func3_ver1(X, Z, task_type):

    # Extract individual features from X and Z matrices
    X1, X2, X3, X4, X5  = X[:,0], X[:,1], X[:,2], X[:,3], X[:,4]
    X6, X7, X8, X9, X10 = X[:,5], X[:,6], X[:,7], X[:,8], X[:,9]
    X11, X12, X13, X14, X15 = X[:,10], X[:,11], X[:,12], X[:,13], X[:,14]
    Z1, Z2, Z3, Z4, Z5 = Z[:,0], Z[:,1], Z[:,2], Z[:,3], Z[:,4]
    Z6, Z7, Z8, Z9, Z10 = Z[:,5], Z[:,6], Z[:,7], Z[:,8], Z[:,9]
    Z11, Z12, Z13, Z14, Z15 = Z[:,10], Z[:,11], Z[:,12], Z[:,13], Z[:,14]

    # Define two-view interactions between X and Z
    interaction1 = + X1 * Z1
    interaction2 = - np.cos(X2 + Z1)
    interaction3 = + np.sqrt(np.exp(X3 + Z1))
    interaction4 = - np.abs(X4 + Z2)
    interaction5 = + np.sin(X4 - Z3)
    interaction6 = - np.abs(X4 * Z4)
    interaction7 = + np.exp(np.abs(X5 + Z5) - 1)
    interaction8 = - 2**(X6 + Z5)
    interaction9 = + np.sin(X7 * np.sin(Z6))
    interaction10 = - np.pi**(X8 * Z7)
    interaction11 = + np.log(np.abs(X8 + Z8) + 1)
    interaction12 = - np.abs(X9 - Z9)
    interaction13 = + np.log(np.exp(X9 + Z10) + 1)
    interaction14 = - np.exp(X10 - Z11)
    interaction15 = + 1/(1 + X11**2 + Z11**2)
    interaction16 = - np.cos(X12 * Z12)
    interaction17 = + np.sqrt(X13**2 + Z12**2)

    # Define within-view effects for X and Z
    X_within_effects = X1 - np.sqrt(np.abs(2 * X2)) - X3 + 1/(1 + X4**2) - np.exp(X5 - X11)  + np.cos(X6) - 2**(X7) + X8 + np.cos(X9 - X15) - np.sqrt(np.exp(X10)) - np.abs(X12) - np.sin(X13 * np.sin(X13)) + np.pi**(X14)
    Z_within_effects = Z1 + np.log(np.abs(Z2)) - Z3 + np.sqrt(np.exp(Z4)) + 1/(1 + Z5**2) - Z6 - np.sin(Z7) + Z8 + Z9 + np.exp(Z10 - Z11) + np.pi**(X12) - Z13 - Z14 * Z15

    # Genereate the linear output
    linear_output = (
        interaction1 + interaction2 + interaction3 + interaction4 + interaction5 +
        interaction6 + interaction7 + interaction8 + interaction9 + interaction10 +
        interaction11 + interaction12 + interaction13 + interaction14 + interaction15 +
        interaction16 + interaction17 + X_within_effects + Z_within_effects
    )

    # Define the ground truth for within-view interactions
    ground_truth = [ (1,1), (2,1), (3,1), (4,2), (4,3), (4,4), (5,5), (6,5), (7,6), (8,7), (8,8), (9,9), (9,10), (10,11), (11,11), (12,12), (13,12)]

    # Generate the response variable Y based on the task type (regression or classification)
    if task_type == "regression":
        Y = linear_output
    elif task_type == "classification":
        Y = generate_binary_response(linear_output)

    return Y, ground_truth


def synth_asym_func4_ver1(X, Z, task_type):

    # Extract individual features from X and Z matrices
    X1, X2, X3, X4, X5  = X[:,0], X[:,1], X[:,2], X[:,3], X[:,4]
    X6, X7, X8, X9, X10 = X[:,5], X[:,6], X[:,7], X[:,8], X[:,9]
    X11, X12, X13, X14, X15 = X[:,10], X[:,11], X[:,12], X[:,13], X[:,14]
    X16, X17, X18, X19, X20 = X[:,15], X[:,16], X[:,17], X[:,18], X[:,19]
    Z1, Z2, Z3, Z4, Z5 = Z[:,0], Z[:,1], Z[:,2], Z[:,3], Z[:,4]
    Z6, Z7, Z8, Z9, Z10 = Z[:,5], Z[:,6], Z[:,7], Z[:,8], Z[:,9]
    Z11, Z12, Z13, Z14, Z15 = Z[:,10], Z[:,11], Z[:,12], Z[:,13], Z[:,14]
    Z16, Z17, Z18, Z19, Z20 = Z[:,15], Z[:,16], Z[:,17], Z[:,18], Z[:,19]

    # Define two-view interactions between X and Z
    interaction1 = + X1 * Z1
    interaction2 = - X2 * Z1
    interaction3 = + X3 * Z1
    interaction4 = - X4 * Z2
    interaction5 = + X4 * Z3
    interaction6 = - X4 * Z4
    interaction7 = + X5 * Z5
    interaction8 = - X6 * Z5
    interaction9 = + X7 * Z6
    interaction10 = - X8 * Z7
    interaction11 = + X8 * Z8
    interaction12 = - X9 * Z9
    interaction13 = + X9 * Z10
    interaction14 = - X10 * Z11
    interaction15 = + X11 * Z11
    interaction16 = - X12 * Z12
    interaction17 = + X13 * Z12

    # Define within-view effects for X and Z
    X_within_effects = X14 - 2*X15 + X16 - X17 + X15 * X18 - X19 * X20
    Z_within_effects = Z13 - Z15 + Z16 - Z17 * Z18 + Z19 - Z14 * Z20

    # Genereate the linear output
    linear_output = (
        interaction1 + interaction2 + interaction3 + interaction4 + interaction5 +
        interaction6 + interaction7 + interaction8 + interaction9 + interaction10 +
        interaction11 + interaction12 + interaction13 + interaction14 + interaction15 +
        interaction16 + interaction17 + X_within_effects + Z_within_effects
    )

    # Define the ground truth for within-view interactions
    ground_truth = [ (1,1), (2,1), (3,1), (4,2), (4,3), (4,4), (5,5), (6,5), (7,6), (8,7), (8,8), (9,9), (9,10), (10,11), (11,11), (12,12), (13,12)]

    # Generate the response variable Y based on the task type (regression or classification)
    if task_type == "regression":
        Y = linear_output
    elif task_type == "classification":
        Y = generate_binary_response(linear_output)

    return Y, ground_truth



def synth_asym_func5_ver2(X, Z, task_type):

    # Extract individual features from X and Z matrices
    X1, X2, X3, X4, X5  = X[:,0], X[:,1], X[:,2], X[:,3], X[:,4]
    X6, X7, X8, X9, X10 = X[:,5], X[:,6], X[:,7], X[:,8], X[:,9]
    X11, X12, X13, X14, X15 = X[:,10], X[:,11], X[:,12], X[:,13], X[:,14]
    X16, X17, X18, X19, X20 = X[:,15], X[:,16], X[:,17], X[:,18], X[:,19]
    Z1, Z2, Z3, Z4, Z5 = Z[:,0], Z[:,1], Z[:,2], Z[:,3], Z[:,4]
    Z6, Z7, Z8, Z9, Z10 = Z[:,5], Z[:,6], Z[:,7], Z[:,8], Z[:,9]
    Z11, Z12, Z13, Z14, Z15 = Z[:,10], Z[:,11], Z[:,12], Z[:,13], Z[:,14]
    Z16, Z17, Z18, Z19, Z20 = Z[:,15], Z[:,16], Z[:,17], Z[:,18], Z[:,19]

    # Define two-view interactions between X and Z
    interaction1 = + X1 * Z1
    interaction2 = - np.abs(X2 * Z1)
    interaction3 = - np.log(np.abs(X3 + Z1))
    interaction4 = + np.sin(X4 * np.sin(Z2))
    interaction5 = - X4 * Z3
    interaction6 = - np.log(np.abs(X4 + Z4))
    interaction7 = + np.sin(X5 * np.sin(Z5))
    interaction8 = + X6 * Z5
    interaction9 = - X7 * Z6
    interaction10 = - np.log(np.abs(X8 + Z7))
    interaction11 = + np.sin(X8 * np.sin(Z8))
    interaction12 = + X9 * Z9
    interaction13 = - np.abs(X9 * Z10)
    interaction14 = + X10 * Z11
    interaction15 = - np.log(np.abs(X11 + Z11))
    interaction16 = + np.sin(X12 * np.sin(Z12))
    interaction17 = + X13 * Z12

    # Define within-view effects for X and Z
    X_within_effects = X14 - np.sin(2 * X15) + X16**2 - X17 + X14 * X18 - np.abs(X19 - X20)
    Z_within_effects = Z13 - 1/(1 + Z14**2 + Z15**2) + Z16 - Z17 * Z18 + Z19 - np.sin(Z20)

    # Genereate the linear output
    linear_output = (
        interaction1 + interaction2 + interaction3 + interaction4 + interaction5 +
        interaction6 + interaction7 + interaction8 + interaction9 + interaction10 +
        interaction11 + interaction12 + interaction13 + interaction14 + interaction15 +
        interaction16 + interaction17 + X_within_effects + Z_within_effects
    )

    # Define the ground truth for within-view interactions
    ground_truth = [ (1,1), (2,1), (3,1), (4,2), (4,3), (4,4), (5,5), (6,5), (7,6), (8,7), (8,8), (9,9), (9,10), (10,11), (11,11), (12,12), (13,12)]

    # Generate the response variable Y based on the task type (regression or classification)
    if task_type == "regression":
        Y = linear_output
    elif task_type == "classification":
        Y = generate_binary_response(linear_output)

    return Y, ground_truth


def synth_asym_func6_ver1(X, Z, task_type):

    # Extract individual features from X and Z matrices
    X1, X2, X3, X4, X5  = X[:,0], X[:,1], X[:,2], X[:,3], X[:,4]
    X6, X7, X8, X9, X10 = X[:,5], X[:,6], X[:,7], X[:,8], X[:,9]
    X11, X12, X13, X14, X15 = X[:,10], X[:,11], X[:,12], X[:,13], X[:,14]
    X16, X17, X18, X19, X20 = X[:,15], X[:,16], X[:,17], X[:,18], X[:,19]
    Z1, Z2, Z3, Z4, Z5 = Z[:,0], Z[:,1], Z[:,2], Z[:,3], Z[:,4]
    Z6, Z7, Z8, Z9, Z10 = Z[:,5], Z[:,6], Z[:,7], Z[:,8], Z[:,9]
    Z11, Z12, Z13, Z14, Z15 = Z[:,10], Z[:,11], Z[:,12], Z[:,13], Z[:,14]
    Z16, Z17, Z18, Z19, Z20 = Z[:,15], Z[:,16], Z[:,17], Z[:,18], Z[:,19]

    # Define two-view interactions between X and Z
    interaction1 = + X1 * Z1
    interaction2 = - np.abs(X2 + Z1)
    interaction3 = + np.sqrt(np.exp(X3 + Z1))
    interaction4 = - np.cos(X4 + Z2)
    interaction5 = + np.sin(X4 - Z3)
    interaction6 = - np.abs(X4 * Z4)
    interaction7 = + np.exp(np.abs(X5 + Z5) - 1)
    interaction8 = - 2**(X6 + Z5)
    interaction9 = + np.sin(X7 * np.sin(Z6))
    interaction10 = - np.pi**(X8 * Z7)
    interaction11 = + np.log(np.abs(X8 + Z8) + 1)
    interaction12 = - np.abs(X9 - Z9)
    interaction13 = + np.log(np.exp(X9 + Z10) + 1)
    interaction14 = - np.sin(X10 + Z11)
    interaction15 = + 1/(1 + X11**2 + Z11**2)
    interaction16 = - np.cos(X12 * Z12)
    interaction17 = + np.sqrt(X13**2 + Z12**2)

    # Define two-view interactions between X and Z
    X_within_effects = X14 - np.sqrt(np.abs(2 * X15)) - X16 + 1/(1 + X17**2) + X18 - np.abs(X15 - X18)  + np.sin(X19 * np.sin(X20))
    Z_within_effects = Z13 + np.log(np.abs(Z15)) - Z16 + np.sin(Z17 + Z18) - np.pi**(Z19) - 1/(1 + Z14**2 + Z20**2)

    # Genereate the linear output
    linear_output = (
        interaction1 + interaction2 + interaction3 + interaction4 + interaction5 +
        interaction6 + interaction7 + interaction8 + interaction9 + interaction10 +
        interaction11 + interaction12 + interaction13 + interaction14 + interaction15 +
        interaction16 + interaction17 + X_within_effects + Z_within_effects
    )

    # Define the ground truth for within-view interactions
    ground_truth = [ (1,1), (2,1), (3,1), (4,2), (4,3), (4,4), (5,5), (6,5), (7,6), (8,7), (8,8), (9,9), (9,10), (10,11), (11,11), (12,12), (13,12)]

    # Generate the response variable Y based on the task type (regression or classification)
    if task_type == "regression":
        Y = linear_output
    elif task_type == "classification":
        Y = generate_binary_response(linear_output)

    return Y, ground_truth
