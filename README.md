# insight_view
尝试追踪pytorch程序运行的流程，获取过程中阶段的关系图与每个步骤的FLOPS


调用方式 

'''python 
flops, macs, params = get_model_profile(model=model, # model
                                    # input_shape=(1,1), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                    args=(x, max_new_tokens, temperature, top_k), # list of positional arguments to the model.
                                    # kwargs=[], # dictionary of keyword arguments to the model.
                                    print_profile=True, # prints the model graph with the measured profile attached to each module
                                    detailed=True, # print the detailed profile
                                    module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                    top_modules=1, # the number of top modules to print aggregated profile
                                    warm_up=1, # the number of warm-ups before measuring the time of each module
                                    as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                    output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                    ignore_modules=None,
                                    mode='generate') # the list of modules to ignore in the profiling


'''
