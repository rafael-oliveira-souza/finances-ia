import ctypes
import tkinter as tk
from enum import Enum
from tkinter import messagebox

import MetaTrader5 as mt5

import RobotGptExecutor

num_candles = 30
qtd_minutes = 20


class TipoSinal(Enum):
    COMPRA = 'COMPRA'
    VENDA = 'VENDA'
    INDEFINIDO = 'INDEFINIDO'


def executar_robot():
    try:
        # Campos básicos
        login = int(entry_login.get())
        numeroRobos = int(entry_robots_number.get())
        robotName = (entry_robot_name.get())
        senha = entry_senha.get()
        servidor = entry_servidor.get()
        symbol = entry_symbol.get()
        volume = float(entry_volume.get())
        timeframe_str = entry_timeframe.get().upper()
        timeframe = getattr(mt5, f"TIMEFRAME_{timeframe_str}")
        # datainicio = normalizar_data(data_inicio.get(), "%d/%m/%Y")
        # datafim = normalizar_data(data_fim.get(), "%d/%m/%Y")

        # Campos adicionais numéricos
        enableMoveStop = (var_move_stop.get())
        valorStopMinimo = int(entry_stop_minimo.get())
        valorPerdaDiaria = int(entry_perda_diaria.get())
        ganhoMaximo = int(entry_ganho_diaria.get())

        # percent_risk = int(entry_percent_risk.get())

        # Criar analisador
        countValue = 0
        # UUIDManager().validarUUID(var_credencial.get())
        # analyzer = CandleAnalyzer(
        #     login=login,
        #     senha=senha,
        #     servidor=servidor,
        #     symbol=symbol,
        #     timeframe=timeframe,
        #     volume=volume,
        #     enableMACD=var_macd.get(),
        #     enableCCI=var_cci.get(),
        #     enableRSI=var_rsi.get(),
        #     enableBollinger=var_bollinger.get(),
        #     enablePatternAnalysis=var_pattern.get(),
        #     enableVerifyDivergence=var_divergence.get(),
        #     enableMartingale=martingale,
        #     stopMinimo=stop_minimo,
        #     maximoPerdaDiaria=perda_diaria,
        #     percentRiskAndProfit=percent_risk
        # )

        # executor = RobotExecutor([analyzer], enableMoveStop=var_move_stop.get(), enableSimulation=var_enable_sim.get(),
        #                          dataInicio=datainicio, dataFim=datafim, maxOrders=entry_max_orders)
        if var_credencial.get() == "":
            raise ValueError("Credencial nao informada!")
        if login == "":
            raise ValueError("Login nao informado!")

        if senha == "":
            raise ValueError("Senha nao informada!")

        if servidor == "":
            raise ValueError("Servidor nao informado!")

        executor = RobotGptExecutor.RobotGptExecutor(hash=var_credencial.get(), robotName=robotName, login=login, senha=senha,
                                    servidor=servidor, symbol=symbol,
                                    timeframe=timeframe, volume=volume,
                                    numberRobots=numeroRobos, enableMoveStop=enableMoveStop,
                                    minimosNegociacao=valorStopMinimo, perdaMaxima=valorPerdaDiaria,
                                    ganhoMaximo=ganhoMaximo)
        executor.execute()
    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro:\n{e}")
        raise ValueError(f"Error: {e}")


# GUI Setup
janela = tk.Tk()
janela.title("Executor de Robô MT5")
janela.geometry("720x600")

# Layout row tracker
linha = 0
coluna = 0


def _impedir_reposo():
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002

    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    )


def criar_label_entrada(texto, valor_default=""):
    global linha, coluna
    tk.Label(janela, text=texto).grid(row=linha, column=coluna, padx=5, pady=5, sticky="w")
    campo = tk.Entry(janela, width=25)
    campo.insert(0, valor_default)
    campo.grid(row=linha + 1, column=coluna, padx=5, pady=5)
    coluna += 1
    if coluna > 2:
        coluna = 0
        linha += 2
    return campo


def criar_checkbox(texto, valor_default=False):
    global linha, coluna
    var = tk.BooleanVar(value=valor_default)
    cb = tk.Checkbutton(janela, text=texto, variable=var)
    cb.grid(row=linha, column=coluna, padx=5, pady=5, sticky="w")
    coluna += 1
    if coluna > 2:
        coluna = 0
        linha += 1
    return var


# Campos de entrada
var_credencial = criar_label_entrada("Chave de Acesso:", "jornada")
entry_robot_name = criar_label_entrada("Nome do Robo:", "Journey Man")
entry_login = criar_label_entrada("Login:", "541072629")
entry_senha = criar_label_entrada("Senha:", "")
entry_servidor = criar_label_entrada("Servidor:", "FTMO-Server4")
entry_symbol = criar_label_entrada("Símbolo:", "XAUUSD")
entry_timeframe = criar_label_entrada("Timeframe (ex: M5):", "M30")
entry_volume = criar_label_entrada("Volume:", "0.02")
entry_robots_number = criar_label_entrada("Numeros de robos:", "10")
# data_inicio = criar_label_entrada("Data Início Simulação:", "01/01/2025")
# data_fim = criar_label_entrada("Data Fim Simulação:", "20/05/2025")

# Checkboxes (3 por linha também)
# var_macd = criar_checkbox("Habilitar MACD", True)
# var_cci = criar_checkbox("Habilitar CCI", True)
# var_rsi = criar_checkbox("Habilitar RSI", True)
# var_bollinger = criar_checkbox("Habilitar Bollinger", True)
# var_pattern = criar_checkbox("Análise de Padrões", True)
# var_divergence = criar_checkbox("Verificar Divergência", True)
# var_enable_sim = criar_checkbox("Habilitar Simulação", False)

# Parâmetros numéricos adicionais
# entry_max_orders = criar_label_entrada("Maximo de Ordens:", "5")
# entry_martingale = criar_label_entrada("Martingale:", "300")
entry_stop_minimo = criar_label_entrada("Mínimos de Negociação:", "300")
entry_perda_diaria = criar_label_entrada("Máx. Perda Diária:", "0")
entry_ganho_diaria = criar_label_entrada("Máx. Ganho Diária:", "1500")
# entry_percent_risk = criar_label_entrada("Percentual Risco/Lucro:", "100")
var_move_stop = criar_checkbox("Habilitar Move Stop", True)

# Botão
linha += 2
tk.Button(
    janela,
    text="Executar Robô",
    command=executar_robot,
    bg="green",
    fg="white",
    height=2,
    width=30
).grid(row=linha, column=0, columnspan=3, pady=20)

janela.mainloop()
