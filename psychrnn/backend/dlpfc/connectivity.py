import numpy as np


def multi_layer_connectivity(
    N_in,
    N_rec,
    N_out,
    N_layers,
    autapses=True,
    dale_ratio=None,
    input=0.3,
    output=0.3,
    rec=0.3,
    ff_ee=0.1,
    ff_ei=0.1,
    fb=0.05,
):

    if type(N_rec) is list:
        if len(N_rec) != N_layers:
            raise Exception("length of N_rec must equal N_layers")
    else:
        N_rec = [N_rec for i in range(N_layers)]

    dale_ratio = dale_ratio if dale_ratio is not None else 1

    if dale_ratio < 0 or dale_ratio > 1:
        raise Exceptions("Dale ratio must be between 0 and 1")

    N_exc = [int(n * dale_ratio) for n in N_rec]
    N_inh = [rec - exc for rec, exc in zip(N_rec, N_exc)]

    # ---------------------------------------
    # generate full input connectivity matrix
    # ---------------------------------------

    input_connectivity = np.zeros((sum(N_rec), N_in))

    # limit input to excitatory neurons in first layer
    input_mat = np.zeros((N_exc[0], N_in))
    input_mat[: int(N_exc[0] * input)] = 1
    for i in range(N_in):
        np.random.shuffle(input_mat[:, i])
    input_connectivity[: N_exc[0]] = input_mat

    # ---------------------------------------
    # generate full recurrent matrix
    # ---------------------------------------

    recurrent_connectivity = np.zeros((sum(N_rec), sum(N_rec)))

    # sparse connectivity within layer
    start_unit = 0
    for i in range(N_layers):

        this_layer_rec = np.zeros((N_rec[i], N_rec[i]))

        for j in range(N_rec[i]):
            this_layer_rec[:int(rec * N_rec[i]), j] = 1
            np.random.shuffle(this_layer_rec[:, j])

        recurrent_connectivity[
            start_unit : (start_unit + N_rec[i]), start_unit : (start_unit + N_rec[i])
        ] = this_layer_rec

        # if N_inh[i] > 0:
        #     # recurrent_connectivity[start_unit:(start_unit+N_rec[i]), (start_unit+N_exc[i]):(start_unit+N_rec[i])] = -1
        #     recurrent_connectivity[
        #         start_unit : (start_unit + N_rec[i]),
        #         (start_unit + N_exc[i]) : (start_unit + N_rec[i]),
        #     ] = 1

        start_unit += N_rec[i]

    # remove autapses
    if not autapses:
        recurrent_connectivity[np.eye(sum(N_rec)) == 1] = 0

    # sparse feedforward and feedback between layers
    start_unit = 0
    for i in range(N_layers - 1):

        # feedforward connections
        ff_ee_mat = np.random.choice(
            [0, 1], size=(N_exc[i + 1], N_exc[i]), p=[1 - ff_ee, ff_ee]
        )
        recurrent_connectivity[
            (start_unit + N_rec[i]) : (start_unit + N_rec[i] + N_exc[i + 1]),
            start_unit : (start_unit + N_exc[i]),
        ] = ff_ee_mat
        if N_inh[i + 1] > 0:
            ff_ei_mat = np.random.choice(
                [0, 1], size=(N_inh[i + 1], N_exc[i]), p=[1 - ff_ei, ff_ei]
            )
            recurrent_connectivity[
                (start_unit + N_rec[i] + N_exc[i + 1]) : (
                    start_unit + N_rec[i] + N_rec[i + 1]
                ),
                start_unit : (start_unit + N_exc[i]),
            ] = ff_ei_mat

        # feedback connections
        fb_mat = np.random.choice([0, 1], size=(N_exc[i], N_exc[i + 1]), p=[1 - fb, fb])
        recurrent_connectivity[
            start_unit : (start_unit + N_exc[i]),
            (start_unit + N_rec[i]) : (start_unit + N_rec[i] + N_exc[i]),
        ] = fb_mat

        start_unit += N_rec[i]

    # ---------------------------------------
    # generate full output matrix
    # ---------------------------------------

    output_connectivity = np.zeros((N_out, sum(N_rec)))
    output_mat = np.zeros((N_out, N_exc[-1]))
    output_mat[:, : int(N_exc[-1] * output)] = 1
    for i in range(N_out):
        np.random.shuffle(output_mat[i])

    if N_inh[-1] > 0:
        output_connectivity[:, -N_rec[-1] : (-N_rec[-1] + N_exc[-1])] = output_mat
    else:
        output_connectivity[:, -N_rec[-1] :] = output_mat

    # ---------------------------------------
    # generate dale rec and dale out matrix
    # ---------------------------------------

    dale_rec = np.zeros((sum(N_rec), sum(N_rec)))
    start_unit = 0
    for i in range(N_layers):
        for j in range(start_unit, start_unit + N_exc[i]):
            dale_rec[j, j] = 1
        start_unit += N_exc[i]
        for k in range(start_unit, start_unit + N_inh[i]):
            dale_rec[k, k] = -1
        start_unit += N_inh[i]

    dale_out = np.zeros((sum(N_rec), sum(N_rec)))
    start_unit = sum(N_rec) - N_rec[-1]
    end_unit = sum(N_rec) - N_inh[-1]
    for i in range(start_unit, end_unit):
        dale_out[i, i] = 1

    # ---------------------------------------
    # return connectivity matrices
    # ---------------------------------------

    return {
        "input_connectivity": input_connectivity,
        "rec_connectivity": recurrent_connectivity,
        "output_connectivity": output_connectivity,
        "Dale_rec": dale_rec,
        "Dale_out": dale_out,
    }
