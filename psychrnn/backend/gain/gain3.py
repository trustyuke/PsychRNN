# Tian added this: add the state with a gain term g = mt + g0

from __future__ import division

from psychrnn.backend.rnnGain import RNN
# from psychrnn.backend.models.basic import Basic
from psychrnn.backend.regularizations import Regularizer
from psychrnn.backend.loss_functions import LossFunction
import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()


class Gain3(RNN):
    """The basic continuous time recurrent neural network model.

    Slightly edited version of the Basic RNN of :class:`psychrnn.backend.models.basic.Basic`

    Basic implementation of :class:`psychrnn.backend.rnn.RNN` with a simple RNN, enabling biological constraints.

    Args:
       params (dict): See :class:`psychrnn.backend.rnn.RNN` for details.

    """

    # def __init__(self, params):

    #     # super(Basic, self).__init__(params)
        

    #     super(RNN, self).__init__(params)
    #     self.output_transfer_function = params.get(
    #         "output_transfer_function", tf.nn.relu
    #     )
    #     self.decision_threshold = params.get("decision_threshold", np.inf)



    def recurrent_timestep(self, rnn_in, state, gainRep):
        """Recurrent time step.

        Given input and previous state, outputs the next state of the network.

        Arguments:
            rnn_in (*tf.Tensor(dtype=float, shape=(?*, :attr:`N_in` *))*): Input to the rnn at a certain time point.
            state (*tf.Tensor(dtype=float, shape=(* :attr:`N_batch` , :attr:`N_rec` *))*): State of network at previous time point.

        Returns:
            new_state (*tf.Tensor(dtype=float, shape=(* :attr:`N_batch` , :attr:`N_rec` *))*): New state of the network.

        """

        new_state = (
            ((1 - self.alpha) * state)
            + self.alpha
            * (
                tf.matmul(self.transfer_function(state+gainRep), self.get_effective_W_rec(), transpose_b=True, name="1")
                + tf.matmul(
                    rnn_in, self.get_effective_W_in(), transpose_b=True, name="2"
                )
                + self.b_rec
            )
            + tf.sqrt(2.0 * self.alpha * self.rec_noise * self.rec_noise)
            * tf.random.normal(tf.shape(input=state), mean=0.0, stddev=1.0)
        )

        return new_state

    def output_timestep(self, state):
        """Returns the output node activity for a given timestep.

        Arguments:
            state (*tf.Tensor(dtype=float, shape=(* :attr:`N_batch` , :attr:`N_rec` *))*): State of network at a given timepoint for each trial in the batch.

        Returns:
            output (*tf.Tensor(dtype=float, shape=(* :attr:`N_batch` , :attr:`N_out` *))*): Output of the network at a given timepoint for each trial in the batch.

        """

        output = (
            tf.matmul(state, self.get_effective_W_out(), transpose_b=True, name="3")
            + self.b_out
        )

        if self.output_transfer_function is not None:
            output = self.output_transfer_function(output)

        return output

    def forward_pass(self):

        """Run the RNN on a batch of task inputs.

        Iterates over timesteps, running the :func:`recurrent_timestep` and :func:`output_timestep`

        Implements :func:`psychrnn.backend.rnn.RNN.forward_pass`.

        Returns:
            tuple:
            * **predictions** (*tf.Tensor(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*) -- Network output on inputs found in self.x within the tf network.
            * **states** (*tf.Tensor(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_rec` *))*) -- State variable values over the course of the trials found in self.x within the tf network.

        """

        threshold_input_mask = tf.zeros((self.N_batch, self.N_in))
        threshold_mask = tf.zeros((self.N_batch, self.N_in), dtype=tf.bool)
        threshold_trial_mask = tf.zeros((self.N_batch, self.N_in), dtype=tf.bool)

        rnn_inputs = tf.unstack(self.x, axis=1)
        state = self.init_state
        rnn_outputs = []
        rnn_states = []
        rnn_inputs_edit = []
        # for rnn_input in rnn_inputs:
        #     this_input = tf.where(threshold_mask, threshold_input_mask, rnn_input)

        #     state = self.recurrent_timestep(this_input, state)
        #     activation = self.transfer_function(state)
        #     output = self.output_timestep(activation)
            
        #     rnn_outputs.append(output)
        #     rnn_states.append(activation)
        #     rnn_inputs_edit.append(this_input)

        #     check_threshold = tf.greater(output, self.decision_threshold)
        #     threshold_trial_mask_vector = tf.expand_dims(
        #         tf.reduce_any(check_threshold, axis=1), axis=1
        #     )
        #     threshold_trial_mask = threshold_trial_mask_vector
        #     for i in range(self.N_in - 1):
        #         threshold_trial_mask = tf.concat(
        #             (threshold_trial_mask, threshold_trial_mask_vector), axis=1
        #         )
        #     threshold_mask = tf.where(
        #         threshold_mask, threshold_mask, threshold_trial_mask
        #     )

        # return (
        #     tf.transpose(a=rnn_outputs, perm=[1, 0, 2]),
        #     tf.transpose(a=rnn_states, perm=[1, 0, 2]),
        #     tf.transpose(a=rnn_inputs_edit, perm=[1, 0, 2]),
        # )


        ####################################################################### Tian edited this
        rnn_gains = tf.unstack(self.g, axis=1)
        threshold_g_mask = tf.zeros((self.N_batch, 1))
        threshold_mask2 = tf.zeros((self.N_batch, 1), dtype=tf.bool)


        for i in range(len(rnn_inputs)):
            rnn_input = rnn_inputs[i]
            rnn_gain = rnn_gains[i]
            # choose whether the input should be zero (choose from threshold input mask when threshold mask equals 1 ) 
            # or actually input (threshold mask equals 0 ) 
            this_input = tf.where(threshold_mask, threshold_input_mask, rnn_input)

            # same rule for gain as input
            this_gain = tf.where(threshold_mask2, threshold_g_mask, rnn_gain)

            # make gain N_batch * N_rec ()
            gainRep = tf.tile(this_gain, [1, self.N_rec])

            state = self.recurrent_timestep(this_input, state, gainRep)

            # two choices to implement the gain: 

            # 1). add gainRep to state: 
            activation = self.transfer_function(state + gainRep)

            # 2). multiply gain to firing rate
            # activation = tf.multiply(self.transfer_function(state), gainRep)
            # or
            # activation = self.transfer_function(tf.multiply(state, gainRep))


            output = self.output_timestep(activation)
            
            rnn_outputs.append(output)
            rnn_states.append(activation)
            rnn_inputs_edit.append(this_input)

            # check_threshold: N_batch * 2 matrix
            check_threshold = tf.greater(output, self.decision_threshold)
            # reduce 2nd dimension of check_threshold: check whether one of the choice reached threshold
            threshold_trial_mask_vector = tf.expand_dims(
                tf.reduce_any(check_threshold, axis=1), axis=1
            )
            # store the vector to threshold_trail_mask 
            threshold_trial_mask = threshold_trial_mask_vector
            # update threshold_mask2 for gain: for current time step, if threshold is reached, threshold_mask2 value update to 1
            threshold_mask2 = tf.where(threshold_mask2, threshold_mask2, threshold_trial_mask)  

            # repeat the threshold_trial_mask for 3 times; now there are 4 columnes, they are all the same
            for i in range(self.N_in - 1):
                threshold_trial_mask = tf.concat(
                    (threshold_trial_mask, threshold_trial_mask_vector), axis=1
                )
            # update threshold_mask for input: for current time step, if threshold is reached, threshold_mask value update to 1
            threshold_mask = tf.where(
                threshold_mask, threshold_mask, threshold_trial_mask
            )

        return (
            tf.transpose(a=rnn_outputs, perm=[1, 0, 2]),
            tf.transpose(a=rnn_states, perm=[1, 0, 2]),
            tf.transpose(a=rnn_inputs_edit, perm=[1, 0, 2]),
        )
        #######################################################################


    def build(self):
        """Build the TensorFlow network and start a TensorFlow session."""
        # --------------------------------------------------
        # Define the predictions
        # --------------------------------------------------
        self.predictions, self.states, self.inputs = self.forward_pass()

        # --------------------------------------------------
        # Define the loss (based on the predictions)
        # --------------------------------------------------
        self.loss = LossFunction(self.params).set_model_loss(self)

        # --------------------------------------------------
        # Define the regularization
        # --------------------------------------------------
        self.reg = Regularizer(self.params).set_model_regularization(self)

        # --------------------------------------------------
        # Define the total regularized loss
        # --------------------------------------------------
        self.reg_loss = self.loss + self.reg

        # --------------------------------------------------
        # Open a session
        # --------------------------------------------------
        self.sess = tf.compat.v1.Session()

        # --------------------------------------------------
        # Record successful build
        # --------------------------------------------------
        self.is_built = True

        return

    # def test(self, trial_batch):
    #     """Test the network on a certain task input.

    #     Arguments:
    #         trial_batch ((*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Task stimulus to run the network on. Stimulus from :func:`psychrnn.tasks.task.Task.get_trial_batch`, or from next(:func:`psychrnn.tasks.task.Task.batch_generator` ).

    #     Returns:
    #         tuple:
    #         * **outputs** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*) -- Output time series of the network for each trial in the batch.
    #         * **states** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_rec` *))*) -- Activity of recurrent units during each trial.
    #     """
    #     if not self.is_built:
    #         self.build()

    #     if not self.is_initialized:
    #         self.sess.run(tf.compat.v1.global_variables_initializer())
    #         self.is_initialized = True

    #     # --------------------------------------------------
    #     # Run the forward pass on trial_batch
    #     # --------------------------------------------------

    #     outputs, states, inputs = self.sess.run(
    #         [self.predictions, self.states, self.inputs],
    #         feed_dict={self.x: trial_batch},
    #     )

    #     return outputs, states, inputs



    def test(self, x, g):
        """Test the network on a certain task input.

        Arguments:
            trial_batch ((*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Task stimulus to run the network on. Stimulus from :func:`psychrnn.tasks.task.Task.get_trial_batch`, or from next(:func:`psychrnn.tasks.task.Task.batch_generator` ).

        Returns:
            tuple:
            * **outputs** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*) -- Output time series of the network for each trial in the batch.
            * **states** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_rec` *))*) -- Activity of recurrent units during each trial.
        """
        if not self.is_built:
            self.build()

        if not self.is_initialized:
            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.is_initialized = True

        # --------------------------------------------------
        # Run the forward pass on trial_batch
        # --------------------------------------------------

        # outputs, states, inputs = self.sess.run(
        #     [self.predictions, self.states, self.inputs],
        #     feed_dict={self.x: trial_batch},
        # )


        ######################################################## Tian edited this
        outputs, states, inputs = self.sess.run([self.predictions, self.states, self.inputs],
                                        feed_dict={self.x: x, self.g: g})
        ########################################################

        return outputs, states, inputs