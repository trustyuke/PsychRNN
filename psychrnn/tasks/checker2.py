from psychrnn.tasks.task import Task
import numpy as np

"""
Edited from the PsychRNN Perceptual Discrimination Task.

Tian Wang Aug.11th

Add another input called gain: 
This is the urgency term multiple with state or firing rate of the RNN later.
For now, this urgency has a linear form: g = g0 + mt
"""


class Checkerboard2AFC(Task):
    """Checkerboard 2AFC task.

    On each trial the network receives four simultaneous inputs
        1a: left tartet red (-1) or green (+1)
        1b: right target red (-1) or green (+1)
        2a: red coherence (-1 to 1) ()
        2b: green coherence (-1 to 1) (G-R) / (R+G)
    The network must determine i) which color has greater coherence and ii) which side that color is on.
        The network outputs two decision variables with one hot encoding (high=1, low=0.2)
        towards the target side representing greater color coherence.

    Args:
        dt (float): The simulation timestep.
        tau (float): The intrinsic time constant of neural state decay.
        T (float): The trial length.
        N_batch (int): The number of trials per training update.
        coherence (float or vector, optional): Green coherence value. Scalar coherence value of range of coherence values (drawn from uniform on each trial)
            Default = [0.5, 1].
        side (float, optional): Probability that right side is green target on given trial.
            Default = 0.5 (random selection).
        noise (float, optional): standard deviation of gaussian noise in stimulus stream
            Default = 0.1
        target_onset (float, optional) : delay before target presentation
            Default = 0.2
        checker_onset (float, optional) : delay before checkerboard presentation
            Default = 0.2
        accumulation_mask (float, optional) : time for accumulation before training
            Default = 0.3

        g0_bound (list with 2 elements): the initial gain range 
        gSlope_bound (list with 2 elements): the slop of gain range  

    """

    def __init__(
        self,
        dt,
        tau,
        T,
        N_batch,
        coherence=[-0.9, 0.9],
        side=0.5,
        noise=0.25,
        target_onset=[250, 500],
        checker_onset=[500, 1000],
        accumulation_mask=300,

        ################################################## Tian added this        
        g0_bound = [0,1],
        gSlope_bound = [0,1],
        ##################################################
    ):

        super().__init__(4, 2, dt, tau, T, N_batch)
        self.coherence = coherence
        self.side = side
        self.noise = noise
        self.target_onset = target_onset
        self.checker_onset = checker_onset
        self.accumulation_mask = accumulation_mask
        self.decision_threshold = 0.7
        self.post_decision_baseline = 0.2

        self.wait = 0.2
        self.hi = 1
        self.lo = 0

        ################################################## Tian added this
        self.g0_bound = g0_bound
        self.gSlope_bound = gSlope_bound
        ##################################################

    def generate_trial_params(self, batch, trial):
        """Define parameters for each trial.

        Implements :func:`~psychrnn.tasks.task.Task.generate_trial_params`.

        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch *batch*.

        Returns:
            dict: Dictionary of trial parameters including the following keys:

            :Dictionary Keys:
                * **coherence** (*float*) -- Probability of a left vs. right flash in given time bin
                * **side** (*int*) -- Either 0 or 1, indicates correct side
                * **noise** (*float*) -- standard deviation the stimlus noise
                * **target_onset** (*float*) -- time before target onset.
                * **checker_onset** (*float*) -- duration of target only (before checker onset).


        """

        params = {}

        params["coherence"] = (
            self.coherence
            if len(self.coherence) == 1
            else np.random.uniform(self.coherence[0], self.coherence[1])
        )
        params["side"] = int(np.random.random() < self.side)
        params["noise"] = self.noise
        params["accumulation_mask"] = self.accumulation_mask
        params["target_onset"] = np.random.randint(self.target_onset[0], self.target_onset[1])
        params["checker_onset"] = np.random.randint(self.checker_onset[0], self.checker_onset[1])

        ################################################## Tian added this
        params["g0_bound"] = self.g0_bound
        params["gSlope_bound"] = self.gSlope_bound
        ##################################################
        return params

    def trial_function(self, t, params, g0, gSlope):
        """Compute the trial properties at :data:`time`.

        Implements :func:`~psychrnn.tasks.task.Task.trial_function`.

        Based on the :data:`params` compute the trial stimulus (x_t), correct output (y_t),
            and mask (mask_t) at :data:`time`.

        Args:
            time (int): The time within the trial (0 <= :data:`time` < :attr:`T`).
            params (dict): The trial params produced by :func:`generate_trial_params`.

        Returns:
            tuple:

            * **x_t** (*ndarray(dtype=float, shape=(2,))*) --
                Trial input at :data:`time` given :data:`params`.
                For ``params['target_onset'] < time < params['target_onset'] + params['stim_duration']`` ,
                1 is added to the noise in both channels, and :data:`params['coherence']`
                is also added in the channel corresponding to :data:`params[dir]`.
            * **y_t** (*ndarray(dtype=float, shape=(2,))*) --
                Correct trial output at :data:`time` given :data:`params`.
                From ``time > params['target_onset'] + params[stim_duration] + 20`` onwards,
                the correct output is encoded using one-hot encoding.
                Until then, y_t is 0 in both channels.
            * **mask_t** (*ndarray(dtype=bool, shape=(*:attr:`N_out` *,))*) --
                True if the network should train to match the y_t,
                False if the network should ignore y_t when training.
                The mask is True for ``time > params['target_onset'] + params['stim_duration']``
                and False otherwise.

        """

        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------

        target_onset = params["target_onset"]
        checker_onset = params["checker_onset"]
        accumulation_mask = params["accumulation_mask"]
        coherence = params["coherence"]
        green_side = params["side"]
        correct_side = green_side if coherence > 0 else abs(green_side - 1)

        # ----------------------------------
        # Generate stimulus
        # ----------------------------------

        x_t = np.zeros(self.N_in)
        if t > target_onset:
            x_t[0] = 2 * green_side - 1
            x_t[1] = -(2 * green_side - 1)
        if t > target_onset + checker_onset:
            x_t[2:] = (params["noise"] ** 2) * np.sqrt(self.dt) * np.random.randn(2)
            x_t[2] += coherence
            x_t[3] -= coherence


        ################################################## Tian added this

        # ----------------------------------
        # Generate gain 
        # ----------------------------------

        g_t = np.zeros(1)
        if t <= target_onset + checker_onset:
            g_t = g0
        if t > target_onset + checker_onset:
            g_t = g0 + gSlope*(t - checker_onset - target_onset) 
        ##################################################

        # ----------------------------------
        # Generate output and mask
        # ----------------------------------

        y_t = np.zeros(self.N_out) + self.wait
        if t > target_onset + checker_onset:
            y_t[correct_side] = self.hi
            y_t[abs(correct_side - 1)] = self.lo

        mask_t = np.ones(self.N_out)
        if (t > target_onset + checker_onset) and (t < target_onset + checker_onset + accumulation_mask):
            mask_t = np.zeros(self.N_out)

        return x_t, y_t, mask_t, g_t



    def generate_trial(self, params):
        """ Loop to generate a single trial.

        Args:
            params(dict): Dictionary of trial parameters generated by :func:`generate_trial_params`.

        Returns:
            tuple:

            * **x_trial** (*ndarray(dtype=float, shape=(*:attr:`N_steps`, :attr:`N_in` *))*) -- Trial input given :data:`params`.
            * **y_trial** (*ndarray(dtype=float, shape=(*:attr:`N_steps`, :attr:`N_out` *))*) -- Correct trial output given :data:`params`.
            * **mask_trial** (*ndarray(dtype=bool, shape=(*:attr:`N_steps`, :attr:`N_out` *))*) -- True during steps where the network should train to match :data:`y`, False where the network should ignore :data:`y` during training.
        """

        ################################################## Tian added this
        g0 = np.random.uniform(params["g0_bound"][0], params["g0_bound"][1])
        gSlope = np.random.uniform(params["gSlope_bound"][0], params["gSlope_bound"][1])
        gainParams = np.array([g0, gSlope])
        ##################################################


        # ----------------------------------
        # Loop to generate a single trial
        # ----------------------------------

        x_data = np.zeros([self.N_steps, self.N_in])
        y_data = np.zeros([self.N_steps, self.N_out])
        mask = np.zeros([self.N_steps, self.N_out])
        g_data = np.zeros([self.N_steps, 1])

        for t in range(self.N_steps):
            x_data[t, :], y_data[t, :], mask[t, :], g_data[t, :] = self.trial_function(t * self.dt, params, g0, gSlope)

        return x_data, y_data, mask, g_data, gainParams


    def batch_generator(self):
        """ Generates a batch of trials.

        Returns:
            Generator[tuple, None, None]:

        Yields:
            tuple:

            * **stimulus** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Task stimuli for :attr:`N_batch` trials.
            * **target_output** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Target output for the network on :attr:`N_batch` trials given the :data:`stimulus`.
            * **output_mask** (*ndarray(dtype=bool, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Output mask for :attr:`N_batch` trials. True when the network should aim to match the target output, False when the target output can be ignored.
            * **trial_params** (*ndarray(dtype=dict, shape =(*:attr:`N_batch` *,))*): Array of dictionaries containing the trial parameters produced by :func:`generate_trial_params` for each trial in :attr:`N_batch`.
        
        """

        batch = 1
        while batch > 0:

            x_data = []
            y_data = []
            mask = []
            params = []
            g_data = []
            gainParams = []
            # ----------------------------------
            # Loop over trials in batch
            # ----------------------------------
            for trial in range(self.N_batch):
                # ---------------------------------------
                # Generate each trial based on its params
                # ---------------------------------------
                p = self.generate_trial_params(batch, trial)
                x,y,m,g,gP = self.generate_trial(p)
                x_data.append(x)
                y_data.append(y)
                mask.append(m)
                params.append(p)
                g_data.append(g)
                gainParams.append(gP)

            batch += 1

            yield np.array(x_data), np.array(y_data), np.array(mask), np.array(params), np.array(g_data), np.array(gainParams)



    def accuracy_function(self, correct_output, test_output, output_mask):
        """Calculates the accuracy of :data:`test_output`.

        Implements :func:`~psychrnn.tasks.task.Task.accuracy_function`.

        Takes the channel-wise mean of the masked output for each trial. Whichever channel has a greater mean is considered to be the network's "choice".

        Returns:
            float: 0 <= accuracy <= 1. Accuracy is equal to the ratio of trials in which the network made the correct choice as defined above.

        """

        chosen_thr = np.where(test_output > 0.6)
        truth_thr = np.where(correct_output > 0.6)

        n_correct = 0
        for i in range(N_batch):
            chosen = chosen_thr[2][chosen_thr[0] == i][0]
            truth = truth_thr[2][truth_thr[0] == i][0]
            n_correct += int(chosen == truth)

        return n_correct / N_batch
