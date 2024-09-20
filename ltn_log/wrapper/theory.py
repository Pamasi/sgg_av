from __future__ import annotations

import dataclasses


import tensorflow as tf
from wandb.integration.keras import WandbMetricsLogger

import ltn
from ltn_log import Formula
from ltn_log.wrapper.constraints import Constraint
from ltn_log.wrapper.grounding import Grounding
from ltn_log.wrapper.domains import Domain, DatasetIterator
from ltn_log.utils.logging.base import MetricsLogger


@dataclasses.dataclass
class Theory:
    constraints: list[Constraint]
    grounding: Grounding
    formula_aggregator: ltn.Wrapper_Formula_Aggregator
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(0.001)
    metrics_loggers: list[MetricsLogger] = dataclasses.field(default_factory=list)
    step: int = 0
    log_every_n_step: int = 50
    agg_sat_metric: tf.keras.metrics.Mean = tf.keras.metrics.Mean("Sat aggregate")
    constraint_metrics: dict[str, tf.keras.metrics.Mean] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.constraint_metrics = {constraint.label: tf.keras.metrics.Mean(constraint.label)
                                   for constraint in self.constraints}

    def train_step_from_domains(
            self,
            constraints_subset: list[Constraint] = None,
            optimizer: tf.keras.optimizers.Optimizer = None
    ) -> None:
        if constraints_subset is not None:
            for constraint in constraints_subset:
                assert (constraint in self.constraints)
        constraints = constraints_subset if constraints_subset is not None else self.constraints
        optimizer = optimizer if optimizer else self.optimizer
        with tf.GradientTape() as tape:
            # wffs = []
            # with concurrent.futures.ThreadPoolExecutor() as executor:
            #     results = [executor.submit(tf.function(lambda: (cstr.call_with_domains(), cstr.label))) for cstr in
            #                constraints]
            #     for res in results:
            #         res, label = res.result()
            #         wffs.append(res)
            #         self.constraint_metrics[label].update_state(wffs[-1].tensor)


            # for cstr in constraints:
            #     wffs.append(cstr.call_with_domains())
            #     self.constraint_metrics[cstr.label].update_state(wffs[-1].tensor)

            # temp = tf.constant([0], dtype=float)
            # temp2 = tf.convert_to_tensor(["N"])

            wffs = []
            for cstr in constraints:
                wffs.append(cstr.call_with_domains())
                # self.constraint_metrics[cstr.label].update_state(wffs[-1].tensor)

            # def fun(k, temp, temp2):
            #     res = constraints[k].call_with_domains()
            #     free = res.free_vars
            #     if not free:
            #         temp2 = tf.concat([temp2, ["N"]], axis=0)
            #     else:
            #         temp2 = tf.concat([temp2, free], axis=0)
            #     return k + 1, tf.concat([temp, [res.tensor]], axis=0), temp2
            #
            # _, temp, temp2 = tf.compat.v1.while_loop(
            #     lambda k, temp, temp2: k < len(constraints),
            #     lambda k, temp, temp2: fun(k, temp, temp2),
            #     [tf.constant(0), temp, temp2],
            #     parallel_iterations=60
            # )
            #
            # temp = temp[1:]
            # temp2 = temp2[1:]
            # temp3 = []
            # for i, cstr in enumerate(constraints):
            #     t = temp2.numpy().astype('str')
            #     temp3.append(Formula(temp[i], t[i] if t[i] != "N" else []))
            #     # self.constraint_metrics[cstr.label].update_state(temp[i])

            # agg_sat = self.formula_aggregator(temp3).tensor
            agg_sat = self.formula_aggregator(wffs).tensor
            loss = 1 - agg_sat
            self.agg_sat_metric.update_state(agg_sat)


        trainable_variables = self.grounding.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
        self.setup_next_minibatches()
        # if (self.step % self.log_every_n_step) == 0:
        #     self.log_metrics()
        self.step += 1

    def log_metrics(self) -> None:
        wandb_loggers = []
        for metric in self.all_metrics:
            for logger in self.metrics_loggers:
                if isinstance(logger, WandbMetricsLogger):
                    wandb_loggers.append(logger)
                    logger.log_value(f"train/{metric.name}", float(metric.result()))
                logger.log_value(metric.name, float(metric.result()), step=self.step)

        for logger in wandb_loggers:
            if hasattr(logger, "commit"):
                logger.commit()

    def reset_metrics(self) -> None:
        for metric in self.all_metrics:
            metric.reset_state()

    def setup_next_minibatches(self) -> None:
        domains: list[Domain] = list(
            {id(dom): dom for cstr in self.constraints for dom in cstr.doms_feed_dict.values()}.values())

        ds_iterators: list[DatasetIterator] = list(
            {id(dom.dataset_iterator): dom.dataset_iterator for dom in domains}.values())
        for ds_iterator in ds_iterators:
            ds_iterator.set_next_minibatch()

    @property
    def all_metrics(self) -> list[tf.keras.metrics.Metric]:
        return [self.agg_sat_metric] + list(self.constraint_metrics.values())
