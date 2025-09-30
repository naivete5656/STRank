def save_result():
    if (toy_type == "_sc") or (toy_type == "_addsc"):
        np.savez(
            f"toy_example/outputs/each_result/{loss_type}_{n_sample}{toy_type}_result.npz",
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            x3=x_test,
            y3=y_test,
            mu1=mu1,
            mu2=mu2,
            mu3=mu3,
            y_test_pred=y_test_pred,
        )
    else:
        np.savez(
            f"toy_example/outputs/each_result/{loss_type}_{n_sample}{toy_type}_{r}_{scale}_result.npz",
            x_train=x_train,
            count_train=count_train,
            x_test=x_test,
            y_test=y_test,
            mu3=mu3,
            y_test_pred=y_test_pred,
        )
