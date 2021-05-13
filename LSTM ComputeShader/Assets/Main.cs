using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

public class Main : MonoBehaviour
{
    LSTMManager.LSTMGroup LSTMGroup;

    public void Start()
    {
        ComputeShader LSTM = LSTMManager.GenerateComputeShader();
        LSTMGroup = LSTMManager.CreateLSTMGroup(new int[] { 2, 2 }, 1);
        LSTMManager.AssignLSTMGroupToShader(LSTMGroup, LSTM);

        LSTM.Dispatch(0, LSTMGroup.Population, 1, 1);
    }

    public void OnApplicationQuit()
    {
        Debug.Log("Disposing");
        LSTMManager.DisposeGroup(LSTMGroup);
    }
}
