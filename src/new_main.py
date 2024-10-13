import time

from Models.BaseModel import BaseModel
from Models.MoneyLineNeuralNetModel001 import MoneyLineNeuralNetModel001
from Models.MoneyLineNeuralNetModel002 import MoneyLineNeuralNetModel002
from Models.XGBModel001 import XGBModel001
from Models.NNModel001 import NNModel001
from ProcessData.InitialDataProcessor import InitialDataProcessor
from data_cuts.all_available_data import AllAvailableData
from data_cuts.base_data_cut import BaseDataCut
from data_cuts.data_cut_n_seasons import DropSeasonsBeforeDate
from data_cuts.drop_old_data import DropOldData
from data_cuts.drop_rank_data import DropRankFeatures
from data_cuts.drop_silly_data import DropSillyData
from experiments.experiment_manager import ExperimentManager

warning_strings = []

cleanup = True
if cleanup:
    em = ExperimentManager()
    em.cleanup_under_performing_models()


def define_models() -> dict[str, BaseModel]:
    # define models
    xgb_model = XGBModel001(params={'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1})
    nn_model = NNModel001(params={'hidden_layer_sizes': (100, 50), 'max_iter': 500, 'alpha': 0.0001})
    og_nn_model = MoneyLineNeuralNetModel001()
    dmb_nn_model_2 = MoneyLineNeuralNetModel002()

    models = {
        xgb_model.get_name(): xgb_model,
        nn_model.get_name(): nn_model,
        og_nn_model.get_name(): og_nn_model,
        dmb_nn_model_2.get_name(): dmb_nn_model_2
    }

    return models


def define_data_cuts(initial_data: InitialDataProcessor) -> dict[str, BaseDataCut]:
    dc_all_data = AllAvailableData(feature_data=initial_data.X,
                                   label_data=initial_data.y,
                                   random_state=42,
                                   test_size=0.2)

    # The output - a dictionary of named BaseDataCuts, starting with "all available data"
    output_data_cuts: dict[str, BaseDataCut] = {dc_all_data.get_name(): dc_all_data}

    # First, remove silly data.
    output_data_cuts = apply_dc_drop_silly_features_to_all_in_list(cuts_list=output_data_cuts)
    output_data_cuts = apply_dc_drop_rank_features_to_all_in_list(cuts_list=output_data_cuts)

    # generate all the season data cuts
    seasonal_cuts: list[DropSeasonsBeforeDate] = []
    for year in range(2021, 2023):
        try:
            season_split = DropSeasonsBeforeDate(feature_data=dc_all_data.get_processed_feature_data(),
                                                 label_data=dc_all_data.get_processed_label_data(),
                                                 season_start_year=year)
            seasonal_cuts.append(season_split)
        except ValueError as drop_season_error:
            if "train set will be empty" in str(drop_season_error):
                print(f"Error: {drop_season_error}")
                continue
            else:
                raise  # Re-raise any other ValueErrors

    # Apply all season datacutting to all existing datacuts
    output_data_cuts = apply_dc_n_seasons_to_all_in_list(seasonal_cuts, output_data_cuts)

    old_cuts: list[DropOldData] = []
    for no_data_before_year in range(2008, 2020):
        try:
            drop_old = DropOldData(feature_data=dc_all_data.get_processed_feature_data(),
                                   label_data=dc_all_data.get_processed_label_data(),
                                   cutoff_year=no_data_before_year)
            old_cuts.append(drop_old)
        except ValueError as error_drop_old:
            if "train set will be empty" in str(error_drop_old):
                print(f"Error: {error_drop_old}")
                continue
            else:
                raise  # Re-raise any other ValueErrors

    output_data_cuts = apply_dc_drop_old_samples_to_all_in_list(old_cuts, output_data_cuts)

    return output_data_cuts


def main():
    # Initialize DataProcessor
    data_processor = InitialDataProcessor("../Data/dataset.sqlite")
    data_processor.load_data()
    data_processor.preprocess_data()

    # Initialize Manager
    mgr = ExperimentManager()

    models = define_models()

    # Iterate through models and add them
    for model in models.values():
        mgr.add_model(model)

    data_cuts = define_data_cuts(initial_data=data_processor)

    # Iterate through data cuts and add them
    for data_cut in data_cuts.values():
        mgr.add_data_cut(data_cut)

    print("Begin training:")
    # Train all models
    results = mgr.train_and_evaluate_all_combinations()
    print("Training complete.")

    # Print results (manager should have transitively logged them anyway)
    for model_name, accuracy in results.items():
        print(f"{model_name} Average Accuracy: {accuracy:.4f}")


def apply_dc_drop_old_samples_to_all_in_list(new_dcs: list[DropOldData], cuts_list: dict[str, BaseDataCut]) \
        -> dict[str, BaseDataCut]:
    output = {}

    for cut in cuts_list.values():
        output[cut.get_name()] = cut
        for new_dc in new_dcs:
            try:
                drop_cut = DropOldData(feature_data=cut.get_processed_feature_data(),
                                       label_data=cut.get_processed_label_data(),
                                       chain_parent=f"{cut.chain_parent} > {cut.get_name()}" if cut.chain_parent else cut.get_name(),
                                       cutoff_year=new_dc.cutoff_year)
                output[f"{cut.get_name()}>{drop_cut.get_name()}"] = drop_cut
            except ValueError as e:
                if "train set will be empty" in str(e):
                    print(f"Error: {e}")
                    continue
                else:
                    raise  # Re-raise any other ValueErrors
    return output


def apply_dc_n_seasons_to_all_in_list(new_dcs: list[DropSeasonsBeforeDate], cuts_list: dict[str, BaseDataCut]) \
        -> dict[str, BaseDataCut]:
    output = {}

    for cut in cuts_list.values():
        output[cut.get_name()] = cut
        for new_dc in new_dcs:
            try:
                drop_cut = DropSeasonsBeforeDate(feature_data=cut.get_processed_feature_data(),
                                                 label_data=cut.get_processed_label_data(),
                                                 chain_parent=f"{cut.chain_parent} > {cut.get_name()}" if cut.chain_parent else cut.get_name(),
                                                 season_start_year=new_dc.season_start_year,
                                                 date_column=new_dc.date_column)
                output[f"{cut.get_name()}>{drop_cut.get_name()}"] = drop_cut
            except ValueError as e:
                if "train set will be empty" in str(e):
                    print(f"Error: {e}")
                    continue
                else:
                    raise  # Re-raise any other ValueErrors
    return output


def apply_dc_drop_rank_features_to_all_in_list(cuts_list: dict[str, BaseDataCut]) \
        -> dict[str, BaseDataCut]:
    """
    Return a new list of DataCuts with all the original values, PLUS a version with their rank columns removed

    :param cuts_list:
    :return:
    """
    output = {}

    for cut in cuts_list.values():
        output[cut.get_name()] = cut
        try:
            drop_cut = DropRankFeatures(feature_data=cut.get_processed_feature_data(),
                                        label_data=cut.get_processed_label_data(),
                                        chain_parent=f"{cut.chain_parent} > {cut.get_name()}" if cut.chain_parent else cut.get_name())
            output[f"{cut.get_name()}>{drop_cut.get_name()}"] = drop_cut
        except ValueError as e:
            if "train set will be empty" in str(e):
                print(f"Error: {e}")
                continue
            else:
                raise  # Re-raise any other ValueErrors
    return output


def apply_dc_drop_silly_features_to_all_in_list(cuts_list: dict[str, BaseDataCut]) \
        -> dict[str, BaseDataCut]:
    """
    Return a new list of DataCuts with all the original values, PLUS a version with their silly columns removed

    :param cuts_list:
    :return:
    """
    output = {}

    for cut in cuts_list.values():
        output[cut.get_name()] = cut
        try:
            serious_cut = DropSillyData(feature_data=cut.get_processed_feature_data(),
                                        label_data=cut.get_processed_label_data(),
                                        chain_parent=f"{cut.chain_parent} > {cut.get_name()}" if cut.chain_parent else cut.get_name())
            output[f"{cut.get_name()}>{serious_cut.get_name()}"] = serious_cut
        except ValueError as e:
            if "train set will be empty" in str(e):
                print(f"Error: {e}")
                continue
            else:
                raise  # Re-raise any other ValueErrors
    return output


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
