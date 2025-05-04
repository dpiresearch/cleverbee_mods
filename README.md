# Cleverbee modifications
The code here is based off of this repo - https://github.com/SureScaleAI/cleverbee

The point of this exercise is to
1. Understand how Cleverbee works
2. Add Llama models to the ecosystem
3. Evaluation research between the models

## Overview

Cleverbee does research through the configuration of three models

1. A Primary model that does the main planning
2. A decision model that decides what to do next
3. A summarization model

All three can be independent - meaning you can have a gemini primary, an anthropic decision model, and a llama summarization model

## Changes

Changes can be classified as follows

### Cleanup
Although the configuration allows choices between Gemini, Anthropic, and local models, the code is primarily geared towards the Google models

This means there is a lot of hardcodes where 'gemini' is used.  The changes here make it more generic and configurable

### Llama

Changes here are related to adding Llama models to the choice of models.  

For the most part the changes are there, however, when it comes to planning, it currently errors out with a bad request.


   
