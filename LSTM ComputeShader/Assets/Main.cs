﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

public class Main : MonoBehaviour
{
    LSTMManager.LSTMGroup LSTMGroup;
    ComputeShader LSTM;
    int Population = 1;
    float[] Inputs;

    public void Start()
    {
        Application.runInBackground = true;
        Inputs = new float[10 * Population];
        LSTM = LSTMManager.GenerateComputeShader();
        LSTMGroup = LSTMManager.CreateLSTMGroup(new int[] { 10, 10, 10, 10, 10 }, Population);
        LSTMManager.AssignLSTMGroupToShader(LSTMGroup, LSTM);
        LSTMGroup.Initialize();

        Debug.Log(LSTMGroup.WeightsBiases.Length);
        for (int i = 0; i < LSTMGroup.WeightsBiases.Length; i++)
        {
            LSTMGroup.WeightsBiases[i] = 1;
        }
        LSTMGroup.SetWeightBiasData();
        
        for (int i = 0; i < Inputs.Length; i++)
        {
            Inputs[i] = 1;
        }

        Debug.Log("Initialize Complete");
    }

    int Count = 0;
    public void FixedUpdate()
    {
        float[] Result = LSTMManager.FeedForward(Inputs, LSTMGroup, LSTM);
        Debug.Log("Update " + Count);
        Debug.Log(String.Join(",", Result));
        //LSTMGroup.DebugData();
        Count++;
    }

    public void OnApplicationQuit()
    {
        Debug.Log("Disposing");
        LSTMManager.DisposeGroup(LSTMGroup);
    }
}
