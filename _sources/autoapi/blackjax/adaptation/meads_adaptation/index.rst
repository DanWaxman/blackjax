blackjax.adaptation.meads_adaptation
====================================

.. py:module:: blackjax.adaptation.meads_adaptation


Classes
-------

.. autoapisummary::

   blackjax.adaptation.meads_adaptation.MEADSAdaptationState


Functions
---------

.. autoapisummary::

   blackjax.adaptation.meads_adaptation.base
   blackjax.adaptation.meads_adaptation.meads_adaptation
   blackjax.adaptation.meads_adaptation.maximum_eigenvalue


Module Contents
---------------

.. py:class:: MEADSAdaptationState



   State of the MEADS adaptation scheme.

   current_iteration
       Current iteration of the adaptation.
   step_size
       Step size for each fold, shape (num_folds,).
   position_sigma
       PyTree with per-fold per-dimension sample standard deviation of the
       position variable, leading axis has size num_folds.
   alpha
       Alpha parameter (momentum persistence) for each fold, shape (num_folds,).
   delta
       Delta parameter (slice translation) for each fold, shape (num_folds,).



   .. py:attribute:: current_iteration
      :type:  int


   .. py:attribute:: step_size
      :type:  blackjax.types.Array


   .. py:attribute:: position_sigma
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: alpha
      :type:  blackjax.types.Array


   .. py:attribute:: delta
      :type:  blackjax.types.Array


.. py:function:: base(num_folds: int = 4, step_size_multiplier: float = 0.5, damping_slowdown: float = 1.0)

   Maximum-Eigenvalue Adaptation of damping and step size for the generalized
   Hamiltonian Monte Carlo kernel :cite:p:`hoffman2022tuning`.

   Full implementation of Algorithm 3 with K-fold cross-chain adaptation and
   chain shuffling. Chains are divided into ``num_folds`` folds; at each step
   statistics from fold ``t mod K`` are used to update the parameters for fold
   ``(t+1) mod K``. Every K steps all chains are reshuffled across folds.

   :param num_folds: Number of folds K to split chains into. Must divide num_chains evenly.
   :param step_size_multiplier: Multiplicative factor applied to the raw step size heuristic (default 0.5
                                as in the paper).
   :param damping_slowdown: Controls the damping floor in early iterations. The floor on γ is
                            ``damping_slowdown / (t·ε)``, so higher values force stronger damping
                            (higher α) in early iterations. Default is 1.0 as in the paper.

   :returns: * *init* -- Function that initializes the warmup state.
             * *update* -- Function that moves the warmup one step forward.


.. py:function:: meads_adaptation(logdensity_fn: Callable, num_chains: int, num_folds: int = 4, step_size_multiplier: float = 0.5, damping_slowdown: float = 1.0, adaptation_info_fn: Callable = return_all_adapt_info) -> blackjax.base.AdaptationAlgorithm

   Adapt the parameters of the Generalized HMC algorithm.

   Full implementation of Algorithm 3 from :cite:p:`hoffman2022tuning` with
   K-fold cross-chain adaptation and periodic chain shuffling.

   Chains are divided into ``num_folds`` folds. At adaptation step ``t``,
   fold ``t mod K`` is frozen (its chains do not advance, Algorithm 3 line 4).
   For each active fold k, the step size is computed from fold ``(k-1) mod K``'s
   preconditioned gradients, and the damping is computed from fold k's own
   positions using that step size. Every K steps all chains are reshuffled
   randomly across folds to prevent fold-assignment bias.

   :param logdensity_fn: The log density probability density function from which we wish to sample.
   :param num_chains: Total number of chains. Must be divisible by ``num_folds``.
   :param num_folds: Number of folds K to split chains into. Default is 4 as in the paper.
   :param step_size_multiplier: Multiplicative factor for the step size heuristic. Default is 0.5 as in
                                the paper.
   :param damping_slowdown: Slows the damping decay relative to the iteration count. Default is 1.0
                            as in the paper. Higher values force stronger damping in early iterations.
   :param adaptation_info_fn: Function to select the adaptation info returned. See return_all_adapt_info
                              and get_filter_adapt_info_fn in blackjax.adaptation.base. By default all
                              information is saved - this can result in excessive memory usage if the
                              information is unused.

   :returns: * *A function that returns the last cross-chain state, a sampling kernel with the*
             * *tuned parameter values (averaged across folds), and all the warm-up states for*
             * *diagnostics.*


.. py:function:: maximum_eigenvalue(matrix: blackjax.types.ArrayLikeTree) -> blackjax.types.Array

   Estimate the largest eigenvalues of a matrix.

   We calculate an unbiased estimate of the ratio between the sum of the
   squared eigenvalues and the sum of the eigenvalues from the input
   matrix. This ratio approximates the largest eigenvalue well except in
   cases when there are a large number of small eigenvalues significantly
   larger than 0 but significantly smaller than the largest eigenvalue.
   This unbiased estimate is used instead of directly computing an unbiased
   estimate of the largest eigenvalue because of the latter's large
   variance.

   :param matrix: A PyTree with equal batch shape as the first dimension of every leaf.
                  The PyTree for each batch is flattened into a one dimensional array and
                  these arrays are stacked vertically, giving a matrix with one row
                  for every batch.


