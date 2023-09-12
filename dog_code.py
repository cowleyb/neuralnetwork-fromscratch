# def gradient_weight2(a1, z2, i):
    #     grad_w = np.zeros(network[1])

    #     for k in range(network[1]):
    #         #y_train[0] needs to be changed to allow for more tests and the x_train above
    #         grad_w[k] = -1 * ((np.exp(z2[(y_train[i]-1)])/np.sum(np.exp(z2))) * a1[k]) - a1[k]
    #     #need a statement adding the grad and averaging it instead of just changing
    #     #gradientw2[(y_train[0]-1)] = grad_w
    #     return grad_w

    # def gradient_bias2(z2, i):
    #     grad_b = -1 * (np.exp(z2[(y_train[i]-1)])/np.sum(np.exp(z2))) - 1
        
    #     #gradientb2[(y_train[0]-1)] = grad_b
    #     return grad_b

    # def gradient_weight1(z1, z2, input_data, parameters, i):
    #     grad_w = np.zeros((network[1], network[0]))

    #     for k in range(network[1]):
    #         for a in range(network[0]):
    #             grad_w[k][a] = (np.exp(z2[(y_train[i]-1)])/np.sum(np.exp(z2))) * parameters["weights2"][(y_train[i]-1), k] * np.where(np.maximum(z1[k], 0) > 0, 1, 0) * input_data[a]
            
    #     #need a statement adding the grad and averaging it instead of just changing
    #     #gradientw1[:] = grad_w
    #     return grad_w

    # def gradient_bias1(z1, z2, parameters, i):
    #     grad_w = np.zeros(network[1])

    #     for k in range(network[1]):
    #             grad_w[k] = (np.exp(z2[(y_train[i]-1)])/np.sum(np.exp(z2))) * parameters["weights2"][(y_train[i]-1), k] * np.where(np.maximum(z1[k], 0) > 0, 1, 0) * 1
            
            
    #     #need a statement adding the grad and averaging it instead of just changing
    #     #gradientb1[:] = grad_w
    #     return grad_w

    # for j in range(100):

    #     gradient_w1 = np.zeros((network[1], network[0]))
    #     gradient_w2 = np.zeros((network[2], network[1]))
    #     gradient_b1 = np.zeros((network[1]))
    #     gradient_b2 = np.zeros((network[2]))
    #     success = 0

    #     for i in range(50):
            
    #         # print(forward(x_train, parameters))
    #         z1, a1, z2, a2 = forward(x_train[:,(i+j*100)], parameters)
    #         # print(forward(x_train[:,0], parameters)[1].shape)
    #         # print(forward(x_train[:,0], parameters)[2].shape)
    #         # print(forward(x_train[:,0], parameters)[3].shape)
            
    #         if i == 1:

    #             gradient_b1[:] = (gradient_bias1(z1, z2, parameters, (i+j*100)))
    #             gradient_w1 = (gradient_weight1(z1, z2, x_train[:,(i+j*100)], parameters, (i+j*100)))
    #             gradient_b2[(y_train[(i+j*100)]-1)] = (gradient_bias2(z2, (i+j*100)))
    #             gradient_w2[(y_train[(i+j*100)]-1)] = (gradient_weight2(a1, z2, (i+j*100)))
            
    #         else:
    #             gradient_b1[:] = ((gradient_bias1(z1, z2, parameters, (i+j*100))) + gradient_b1[:])/2
    #             gradient_w1[:] = ((gradient_weight1(z1, z2, x_train[:,(i+j*100)], parameters, (i+j*100))) + gradient_w1[:])/2
    #             gradient_b2[(y_train[(i+j*100)]-1)] = ((gradient_bias2(z2, (i+j*100))) + gradient_b2[(y_train[(i+j*100)]-1)])/2
    #             gradient_w2[(y_train[(i+j*100)]-1)] = ((gradient_weight2(a1, z2, (i+j*100))) + gradient_w2[(y_train[(i+j*100)]-1)])/2

    #         a2 = list(a2)
    #         if ((a2.index(max(a2)))+1) == y_train[(i+j*100)]:
    #             success = success + 1

    #     success = (success/50)*100

    #     print(f'Success Rate {success} %')

    #     gradients = {}
    #     gradients['weights1'] = gradient_w1
    #     gradients['weights2'] = gradient_w2
    #     gradients['bias1'] = gradient_b1
    #     gradients['bias2'] = gradient_b2

    #     learning_rate = 0.1

    #     update_parameters(parameters, gradients, learning_rate)
    
    