
def generateDataset(video_ids, segment_table, num_windows = 1):
    logits, states = [], []
    for idx, video_id in enumerate(video_ids):
        if not checkParquetExist("v_"+video_id): 
            print(f"passed: index = {idx}, id =  {video_id}")
            continue
        logits_states = videos_to_logits_states("v_"+video_id, segment_table)
        total_frames = len(logits_states[1])
        indices = np.arange(total_frames)
        indices_to_fetch = np.array([np.clip(np.arange(index - num_windows // 2, index + num_windows // 2 + 1), 0, total_frames - 1) for index in indices])
        logits += [*np.array(logits_states[0])[indices_to_fetch]]
        states += [*np.array(list(map(lambda s: s.value, logits_states[1])))[indices_to_fetch]]
        print(f"generated index = {idx}, id =  {video_id}")
    return LogitDataset(logits, states)