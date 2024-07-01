A package to design RNA with Reinforcement Learning.

call with
python -m src.optimization.training

Arguments:
--result_dir: where to write the results
--input_file: file that contains the input data in fasta format
--num_episodes: the episodes for the agent to learn
--masked: set to true, if parts of the input data are masked (partial RNA design)
--agent_file: file to the pretrained agent
--config_file: where the config is located
--config: the config to use
--save_path: the path to save the agent in