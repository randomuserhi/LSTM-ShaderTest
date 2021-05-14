using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

public class _LSTM
{
    public static Random R = new Random();

    protected Matrix[] Nodes;
    protected Matrix[] CellStates;

    protected Matrix[] InputWeights;
    protected Matrix[] InputStateWeights;
    protected Matrix[] InputBias;
    protected Matrix[] InputModulationWeights;
    protected Matrix[] InputModulationStateWeights;
    protected Matrix[] InputModulationBias;
    protected Matrix[] ForgetWeights;
    protected Matrix[] ForgetStateWeights;
    protected Matrix[] ForgetBias;
    protected Matrix[] OutputWeights;
    protected Matrix[] OutputStateWeights;
    protected Matrix[] OutputBias;

    public _LSTM(int[] Structure)
    {
        Nodes = new Matrix[Structure.Length];
        CellStates = new Matrix[Structure.Length - 1];
        InputWeights = new Matrix[Structure.Length - 1];
        InputStateWeights = new Matrix[Structure.Length - 1];
        InputBias = new Matrix[Structure.Length - 1];
        InputModulationWeights = new Matrix[Structure.Length - 1];
        InputModulationStateWeights = new Matrix[Structure.Length - 1];
        InputModulationBias = new Matrix[Structure.Length - 1];
        ForgetWeights = new Matrix[Structure.Length - 1];
        ForgetStateWeights = new Matrix[Structure.Length - 1];
        ForgetBias = new Matrix[Structure.Length - 1];
        OutputWeights = new Matrix[Structure.Length - 1];
        OutputStateWeights = new Matrix[Structure.Length - 1];
        OutputBias = new Matrix[Structure.Length - 1];
        for (int i = 1, j = 0; i < Structure.Length; i++, j++)
        {
            Nodes[j] = new Matrix(1, Structure[j]);
            CellStates[j] = new Matrix(1, Structure[i]);

            Nodes[j].SetData();
            CellStates[j].SetData();

            InputWeights[j] = new Matrix(Structure[j], Structure[i]);
            InputStateWeights[j] = new Matrix(Structure[i], Structure[i]);
            InputBias[j] = new Matrix(1, Structure[i]);
            InitializeWeights(InputWeights[j], InputStateWeights[j], InputBias[j]);
            InputModulationWeights[j] = new Matrix(Structure[j], Structure[i]);
            InputModulationStateWeights[j] = new Matrix(Structure[i], Structure[i]);
            InputModulationBias[j] = new Matrix(1, Structure[i]);
            InitializeWeights(InputModulationWeights[j], InputModulationStateWeights[j], InputModulationBias[j]);
            ForgetWeights[j] = new Matrix(Structure[j], Structure[i]);
            ForgetStateWeights[j] = new Matrix(Structure[i], Structure[i]);
            ForgetBias[j] = new Matrix(1, Structure[i]);
            InitializeWeights(ForgetWeights[j], ForgetStateWeights[j], ForgetBias[j]);
            OutputWeights[j] = new Matrix(Structure[j], Structure[i]);
            OutputStateWeights[j] = new Matrix(Structure[i], Structure[i]);
            OutputBias[j] = new Matrix(1, Structure[i]);
            InitializeWeights(OutputWeights[j], OutputStateWeights[j], OutputBias[j]);
        }
        Nodes[Structure.Length - 1] = new Matrix(1, Structure[Structure.Length - 1]);
        Nodes[Structure.Length - 1].SetData();
    }

    public _LSTM(int[] Structure, _LSTM Copy)
    {
        Nodes = new Matrix[Structure.Length];
        CellStates = new Matrix[Structure.Length - 1];
        InputWeights = new Matrix[Structure.Length - 1];
        InputStateWeights = new Matrix[Structure.Length - 1];
        InputBias = new Matrix[Structure.Length - 1];
        InputModulationWeights = new Matrix[Structure.Length - 1];
        InputModulationStateWeights = new Matrix[Structure.Length - 1];
        InputModulationBias = new Matrix[Structure.Length - 1];
        ForgetWeights = new Matrix[Structure.Length - 1];
        ForgetStateWeights = new Matrix[Structure.Length - 1];
        ForgetBias = new Matrix[Structure.Length - 1];
        OutputWeights = new Matrix[Structure.Length - 1];
        OutputStateWeights = new Matrix[Structure.Length - 1];
        OutputBias = new Matrix[Structure.Length - 1];
        for (int i = 1, j = 0; i < Structure.Length; i++, j++)
        {
            Nodes[j] = new Matrix(1, Structure[j]);
            CellStates[j] = new Matrix(1, Structure[i]);

            Nodes[j].SetData();
            CellStates[j].SetData();

            InputWeights[j] = new Matrix(Copy.InputWeights[j]);
            InputStateWeights[j] = new Matrix(Copy.InputStateWeights[j]);
            InputBias[j] = new Matrix(Copy.InputBias[j]);
            InputModulationWeights[j] = new Matrix(Copy.InputModulationWeights[j]);
            InputModulationStateWeights[j] = new Matrix(Copy.InputModulationStateWeights[j]);
            InputModulationBias[j] = new Matrix(Copy.InputModulationBias[j]);
            ForgetWeights[j] = new Matrix(Copy.ForgetWeights[j]);
            ForgetStateWeights[j] = new Matrix(Copy.ForgetStateWeights[j]);
            ForgetBias[j] = new Matrix(Copy.ForgetBias[j]);
            OutputWeights[j] = new Matrix(Copy.OutputWeights[j]);
            OutputStateWeights[j] = new Matrix(Copy.OutputStateWeights[j]);
            OutputBias[j] = new Matrix(Copy.OutputBias[j]);
        }
        Nodes[Structure.Length - 1] = new Matrix(1, Structure[Structure.Length - 1]);
        Nodes[Structure.Length - 1].SetData();
    }

    private void InitializeWeights(Matrix Weights, Matrix StateWeights, Matrix Bias)
    {
        for (int k = 0; k < Weights.Buffer.Length; k++)
        {
            Weights.Buffer[k] = 1;// ((float)R.NextDouble() * 2 - 1);
        }

        for (int k = 0; k < StateWeights.Buffer.Length; k++)
        {
            StateWeights.Buffer[k] = 1;// ((float)R.NextDouble() * 2 - 1);
        }

        for (int k = 0; k < Bias.Buffer.Length; k++)
        {
            Bias.Buffer[k] = 1;// ((float)R.NextDouble() * 2 - 1);
        }

        Weights.SetData();
        StateWeights.SetData();
        Bias.SetData();
    }

    public float[] FeedForward(float[] Input)
    {
        Nodes[0].CopyValues(Input);
        for (int j = 1, i = 0; j < Nodes.Length; j++, i++)
        {
            //TODO:: optimze by combining all these matrices into 1 2x2 matrix to perform in one operation similar to CNN optimization for convolutions
            Matrix InputGate = Nodes[i] * InputWeights[i] + Nodes[j] * InputStateWeights[i] + InputBias[i];
            Matrix InputModulationGate = Nodes[i] * InputModulationWeights[i] + Nodes[j] * InputModulationStateWeights[i] + InputModulationBias[i];
            Matrix ForgetGate = Nodes[i] * ForgetWeights[i] + Nodes[j] * ForgetStateWeights[i] + ForgetBias[i];
            Matrix OutputGate = Nodes[i] * OutputWeights[i] + Nodes[j] * OutputStateWeights[i] + OutputBias[i];

            InputGate.SigmoidActivation();
            InputModulationGate.SigmoidActivation();
            ForgetGate.TanhActivation();
            OutputGate.SigmoidActivation();

            Matrix.CWiseMultiply(CellStates[i], InputGate, CellStates[i]);
            CellStates[i] += Matrix.CWiseMultiply(ForgetGate, InputModulationGate, ForgetGate);

            Matrix CellState = new Matrix(CellStates[i]);
            CellState.TanhActivation();

            Matrix.CWiseMultiply(OutputGate, CellState, Nodes[j]);
        }
        return Nodes[Nodes.Length - 1].GetData();
    }

    public void Reset()
    {
        for (int i = 0; i < CellStates.Length; i++)
        {
            CellStates[i].SetZero();
            Nodes[i].SetZero();
        }
        Nodes[Nodes.Length - 1].SetZero();
    }
}
