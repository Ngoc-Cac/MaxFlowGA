from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk, networkx as nx, algorithms as alg

# import logging as lg

crossover_func: int = 1
pop_size: int = 500
mutation_rate: int = 0.05
max_iter: int = 200
best_max_iter: int = 50

edit_instructions = \
"""► Nút Add Vertex:
    Cú pháp: '(x, y)'
    Lưu ý: Vị trí của đỉnh s được chọn làm gốc (0, 0). Đỉnh t được cho trước tại (0.5, 0)
    Ví dụ: '(1, 0)' → Thêm đỉnh có tọa độ (1, 0)

► Nút Add Edge:
    Cú pháp: '(v1, v2): weight'
    Ví dụ: `(s, 1): 10` → Thêm cạnh từ đỉnh s đến đỉnh 1 với trọng số là 10

► Nút Remove Vertex:
    Cú pháp: 'vi'
    Ví dụ: '2' → Bỏ đỉnh 2 và các cạnh tương ứng
    Lưu ý: Không cho phép bỏ đỉnh s và đỉnh t

► Nút Remove Edge:
    Cú pháp: '(v1, v2)'
    Ví dụ: '(s, 1)' → Bỏ cạnh từ đỉnh s đến đỉnh 1.

► Nút Clear: Xóa hết tất cả các đỉnh và cạnh trong đồ thị, đưa đồ thị về trạng thái ban đầu"""

f1_meta_information = \
"""► Population Size: Tổng số lượng cá thể trong quần thể, biểu thị số lượng giải pháp được xem xét trong mỗi thế hệ của thuật toán di truyền.

► Mutation Rate: Tỷ lệ đột biến, chỉ tỷ lệ xác suất mà các cá thể trong quần thể sẽ trải qua sự thay đổi ngẫu nhiên.

► Max Generation: Số lượng thế hệ tối đa mà thuật toán di truyền sẽ chạy trước khi dừng lại.

► Crossover Function: Hàm lai ghép, xác định cách thức kết hợp giữa hai cá thể để tạo ra cá thể mới.

► Ford Fulkerson: Chạy thuật toán Ford Fulkerson để tìm luồng cực đại trong đồ thị. Nhấn nút này để bắt đầu chạy thuật toán Ford Fulkerson.

► Genetic Algorithm: Chạy thuật toán di truyền để tìm kiếm giải pháp. Nhấn nút này để bắt đầu chạy thuật toán di truyền.

► Set Population Size: Thiết lập kích thước của quần thể trong mỗi thế hệ. Nhấn nút này để nhập giá trị (một số nguyên dương).

► Set Mutation Rate: Thiết lập tỉ lệ đột biến cho mỗi gen trong một cá thể. Nhấn nút này để nhập giá trị (một số thực trong khoảng 0 đến 1).

► Set Max Generation: Thiết lập số lượng thế hệ tối đa mà thuật toán sẽ chạy. Nhấn nút này để nhập giá trị (một số nguyên dương).

► Change Crossover Function: Thay đổi hàm lai ghép để kết hợp các cặp cá thể tạo ra cá thể con.
    Số `1` thể hiện phương thức lai tạo ưu tiên bảo toàn luồng.
    Số `2` thể hiện phương thức lai tạo một điểm.
    Số `3` thể hiện phương thức lai tạo một điểm.

► Toggle Update: Thực hiện cập nhật cá thể tốt nhất trong lúc chạy."""

f1_information = \
f"""Crossover Function In Use: {crossover_func}

Population Size: {pop_size}

Mutation Rate: {mutation_rate}

Max Generations: {max_iter}"""

graph: dict[int, list[int]] = {'s': [],
                               't': []}
main_nodes_pos: dict[int, tuple[int, int]] = {'s': (0, 0),
                                              't': (0.5, 0)}
capacity_matrix: list[list[int]] = [[0, 0],
                                    [0, 0]]
#!!!Demo Graph!!!
# main_nodes_pos: dict[int: tuple[int, int]] =  {'s': (0, 0),
#                                          1: (0.5, 0.5),
#                                          2: (0.5, -0.5),
#                                          3: (1, 0.5),
#                                          4: (1, -0.5),
#                                          't': (1.5, 0)}
# graph: dict[int: tuple[int]] = {'s': (1, 2),
#                                 1: (3,),
#                                 2: (1, 4),
#                                 3: ('t',),
#                                 4: (3, 't'),
#                                 't': ()}
# capacity_matrix: list[list[int]] = [[0,16,13, 0, 0, 0],
#                              [0, 0, 0,12, 0, 0],
#                              [0, 4, 0, 0,14, 0],
#                              [0, 0, 0, 0, 0,20],
#                              [0, 0, 0, 7, 0, 4],
#                              [0, 0, 0, 0, 0, 0]]

root = tk.Tk()
root.geometry('1200x650+100+100')
root.title("A Genetic Take On Maximum Flow")
root.resizable(False, False)
root.configure(bg='#ebf7ff')

# Graph's positions
f1_main_graph = (0, 0.1)
f1_ff_graph = (0.25, 0.1)
f1_best_gen_graph = (0.5, 0.1)
# f1_best_curgen_graph = (0.75, 0.1)
f2_graph = (0.65, 0.16)

info_box_offset = (0.062, 0.6)
graph_name_offset = (0.08, 0.02 + 0.02)

# Track the visibility state of the instruction label and button state
instruction_visible = False
button_toggled = False
DoUpdate = False
PrematureStop = True

#Def Graph
def generate_edges(adjacent_nodes: dict[int, list[int]]) -> tuple[tuple[int, int]]:
    return tuple((node, adjacent) for node, adjacents in adjacent_nodes.items()
                                   for adjacent in adjacents)
def give_edge_weights(edges: tuple[tuple[int, int]], weight_matrix: list[list[int]])\
                    -> dict[tuple[int, int]: int]:
    return {edge: weight_matrix[edge[0] if edge[0] != 's' else 0][edge[1] if edge[1] != 't' else -1] for edge in edges}
def draw_graph(master, graph: nx.Graph, nodes_pos: dict[int, tuple[float | int]],
                       edges: tuple[tuple[int, int]], edge_weights: dict[tuple[int, int]: int],
                       pos: tuple[float], *,
                       canvas: FigureCanvasTkAgg | None = None) -> None | FigureCanvasTkAgg:
    fig = Figure((3,3), dpi=100)
    g = fig.add_subplot()
    nx.draw_networkx_nodes(graph, nodes_pos, label=True, ax=g)
    nx.draw_networkx_labels(graph, nodes_pos, ax=g)
    nx.draw_networkx_edges(graph,nodes_pos,edges, ax=g)
    nx.draw_networkx_edge_labels(graph, nodes_pos, edge_weights, ax=g)
    if canvas:
        canvas.figure = fig
        canvas.draw()
    else:
        canvas = FigureCanvasTkAgg(fig, master=master)
        canvas.get_tk_widget().place(relx=pos[0], rely=pos[1])
        return canvas

def toggle_instruction_frame():
    global instruction_visible, button_toggled
    if instruction_visible and active_button == 'button1':
        instruc_lb1.grid_remove()
    elif active_button == 'button1':
        instruc_lb1.grid()
    instruction_visible = not instruction_visible
    
    # Toggle button color and text color
    if button_toggled:
        instruc_btn.config(bg='white', fg='black')
    else:
        instruc_btn.config(bg='blue', fg='white')
    button_toggled = not button_toggled
    root.update()

def show_frame1():
    global active_button
    active_button = 'button1'
    main_frame2.grid_remove()
    edit_fm2.grid_remove()
    main_frame1.grid()
    edit_fm1.grid()
    draw_graph(main_frame1, nx.DiGraph(graph), main_nodes_pos, edges,
               give_edge_weights(edges, capacity_matrix), f1_main_graph,
               canvas=main_graph_canvas)
    
def show_frame2():
    global active_button
    if instruction_visible:
        toggle_instruction_frame()
    active_button = 'button2'
    main_frame1.grid_remove()
    edit_fm1.grid_remove()
    main_frame2.grid()
    edit_fm2.grid()

def AddEdgeInput(master: tk.Tk | tk.Frame, entryWidget: tk.Entry) -> None:
    error: bool = False
    input: str | list[str] = entryWidget.get()
    if not input:
        print("No input was given")
        error = True
    elif (num := input.count(':')) != 1:
        print(f"Syntax error: {num} ':' was given, expected one")
    elif (num := input.count(',')) != 1:
        print(f"Syntax error: {num} ',' was given, expected one")
    elif (num := input.count('(')) != 1:
        print(f"Syntax error: {num} '(' was given, expected one")
    elif (num := input.count(')')) != 1:
        print(f"Syntax error: {num} ')' was given, expected one")
    
    if error:
        return

    input = input.replace(' ', '').split(':')
    edge: list[str] = input[0].split(',')
    
    first_vertex: str | int = edge[0].split('(')
    second_vertex: str | int = edge[1].split(')')

    if len(first_vertex) != 2 or not first_vertex[1]:
        print("Node does not exist")
        return
    elif len(second_vertex) != 2 or not second_vertex[0]:
        print("Node does not exist")
        return
    
    if first_vertex[1].lower() == 's':
        first_vertex = 's'
    else:
        try:
            first_vertex = int(first_vertex[1])
        except ValueError:
            print("Node must be a number or literals of either 's' or 't'")
            return
    if second_vertex[0].lower() == 't':
        second_vertex = 't'
    else:
        try:
            second_vertex = int(second_vertex[0])
        except ValueError:
            print("Node must be a number or literals of either 's' or 't'")
            return

    try:
        weight = int(input[1])
    except ValueError:
        print("Capacity must be a number")
        return

    if not first_vertex in graph:
        print("Node does not exist")
        return
    if not second_vertex in graph:
        print("Node does not exist")
        return

    graph[first_vertex].append(second_vertex)
    capacity_matrix[first_vertex if first_vertex != 's' else 0][second_vertex if second_vertex != 't' else -1] = weight

    entry2.delete(0,'end')
    edges = generate_edges(graph)
    draw_graph(master, nx.DiGraph(graph), main_nodes_pos, edges, give_edge_weights(edges, capacity_matrix), f2_graph,
               canvas=edit_graph_canvas)
def AddVertexInput(master: tk.Tk | tk.Frame, entryWidget: tk.Entry) -> None:
    error: bool = False
    input: str | list[str] = entryWidget.get().replace(' ', '')
    if not input:
        print("No input was given")
        error = True
    elif not input.replace(',', '').replace('(', '').replace(')','').replace('.','').replace('-', '').isnumeric():
        print(f"Syntax error: unrecognised character found")
        error = True
    elif (num := input.count(',')) != 1:
        print(f"Syntax error: {num} ',' was given, expected one")
        error = True
    elif (num := input.count('(')) != 1:
        print(f"Syntax error: {num} '(' was given, expected one")
        error = True
    elif (num := input.count(')')) != 1:
        print(f"Syntax error: {num} ')' was given, expected one")
        error = True

    if error:
        return
    else:
        error = False
    
    input = input.split(',')

    x_coords = input[0].split('(')
    y_coords = input[1].split(')')

    if len(x_coords) != 2:
        print("Something went wrong")
        error = True
    if len(y_coords) != 2:
        print("Something went wrong")
        error = True
    
    if error:
        return
    
    try:
        x_coords = float(x_coords[1])
        y_coords = float(y_coords[0])
    except (ValueError, TypeError):
        print("Coordinates must be numeric")
        return
    
    if len(graph) == len(capacity_matrix):
        vertex_no = len(graph) - 1
        for row in capacity_matrix:
            row.insert(-1,0)
        capacity_matrix.append([0] * len(capacity_matrix[0]))
    else:
        keys = [key for key in graph if key != 's' and key != 't']
        keys.extend([0, len(capacity_matrix) - 1])
        keys.sort()
        for i, key in enumerate(keys):
            if keys[i + 1] - key != 1:
                vertex_no = i + 1
                break
    graph[vertex_no] = []
    main_nodes_pos[vertex_no] = (x_coords, y_coords)
    if x_coords >= main_nodes_pos['t'][0]:
        main_nodes_pos['t'] = (x_coords + 0.5, main_nodes_pos['t'][1])
    if x_coords <= main_nodes_pos['s'][0]:
        main_nodes_pos['s'] = (x_coords - 0.5, main_nodes_pos['s'][1])

    entry2.delete(0, 'end')
    edges = generate_edges(graph)
    draw_graph(master, nx.DiGraph(graph), main_nodes_pos, edges, give_edge_weights(edges, capacity_matrix), f2_graph,
               canvas=edit_graph_canvas)

def RemoveVertex(master: tk.Tk | tk.Frame, entryWidget: tk.Entry) -> None:
    error: bool = False
    input = entryWidget.get()

    try:
        input = int(input)
    except ValueError:
        print("Input must be numerical")
        error = True
    except TypeError:
        print("Could not handle whatever was given")
        error = True
    
    if error:
        return
    
    if input in graph:
        del graph[input]
        del main_nodes_pos[input]
        capacity_matrix[input] = [0] * len(capacity_matrix)
        for row in capacity_matrix:
            row[input] = 0
        for _, val in graph.items():
            if input in val:
                val.remove(input)
    
    entry2.delete(0, 'end')
    edges = generate_edges(graph)
    draw_graph(master, nx.DiGraph(graph), main_nodes_pos, edges, give_edge_weights(edges, capacity_matrix), f2_graph,
               canvas=edit_graph_canvas)
def RemoveEdge(master: tk.Tk | tk.Frame, entryWidget: tk.Entry)-> None:
    error: bool = False
    input: str | list[str] = entryWidget.get().replace(' ', '')
    if not input:
        print("No input was given")
        error = True
    elif not input.replace('(', '').replace(')', '').replace(',', '').isalnum():
        print("Syntax error: unregcognised character found")
        error = True
    elif (num := input.count(',')) != 1:
        print(f"Syntax error: {num} ',' was given, expected one")
        error = True
    elif (num := input.count('(')) != 1:
        print(f"Syntax error: {num} '(' was given, expected one")
        error = True
    elif (num := input.count(')')) != 1:
        print(f"Syntax error: {num} ')' was given, expected one")
        error = True
    
    if error:
        return

    input = input.split(',')

    first_vertex = input[0].split('(')
    second_vertex = input[1].split(')')

    if len(first_vertex) != 2:
        print("Something went wrong")
        error = True
    if len(second_vertex) != 2:
        print("Something went wrong")
        error = True
    
    if error:
        return
    
    if first_vertex[1].lower() == 's':
        first_vertex = 's'
    else:
        try:
            first_vertex = int(first_vertex[1])
        except ValueError:
            print("Node must be a number or literals of either 's' or 't'")
            return
        except TypeError:
            print("Missing first vertex")
            return
    if second_vertex[0].lower() == 't':
        second_vertex = 't'
    else:
        try:
            second_vertex = int(second_vertex[0])
        except ValueError:
            print("Node must be a number or literals of either 's' or 't'")
            return
        except TypeError:
            print("Missing second vertex")
            return
    
    if not first_vertex in graph or not second_vertex in graph:
        print("Vertex is not in graph")
        return

    try:
        graph[first_vertex].remove(second_vertex)
    except ValueError:
        print("Edge is not in graph")
        return

    entry2.delete(0, 'end')
    capacity_matrix[first_vertex if first_vertex != 's' else 0][second_vertex if second_vertex != 't' else -1] = 0
    edges = generate_edges(graph)
    draw_graph(master, nx.DiGraph(graph), main_nodes_pos, edges, give_edge_weights(edges, capacity_matrix), f2_graph,
               canvas=edit_graph_canvas)
def ClearGraph(master: tk.Tk | tk.Frame) -> None:
    global graph, main_nodes_pos, capacity_matrix
    graph = {'s': [],
             't': []}
    main_nodes_pos = {'s': (0, 0),
                      't': (0.5, 0)}
    capacity_matrix = [[0, 0],
                       [0, 0]]
    edges = generate_edges(graph)
    draw_graph(master, nx.DiGraph(graph), main_nodes_pos, edges, give_edge_weights(edges, capacity_matrix), f2_graph,
               canvas=edit_graph_canvas)

def RunFordFulkerson(master: tk.Tk | tk.Frame) -> None:
    if not sum(capacity_matrix[0]) or not sum([i[-1] for i in capacity_matrix]):
        raise Exception("Invalid graph")
    disablef1butts()

    result = alg.ford_fulkerson(capacity_matrix)
    flow = alg.assign_flow(capacity_matrix, result[1])
    edges = generate_edges(graph)
    weights = give_edge_weights(edges, flow)
    draw_graph(master, nx.DiGraph(graph), main_nodes_pos, edges, weights, f1_ff_graph, canvas=ff_graph_canvas)
    ff_info.config(text=f"""Fitness score: {alg.fitness(capacity_matrix, flow): .4f}
Max Flow: {alg.col_sum(flow, -1)}""")
    
    enablef1butts()

def RunGenAlg(master: tk.Tk | tk.Frame) -> None:
    if not sum(capacity_matrix[0]) or not sum([i[-1] for i in capacity_matrix]):
        raise Exception("Invalid graph")
    disablef1butts()

    result = alg.max_flow_GA(capacity_matrix, crossover_func,
                             pop_size=pop_size, mutation_rate=mutation_rate,
                             max_iter=max_iter,
                             best_max_iter=best_max_iter if PrematureStop else None,
                             update_procedure=lambda x, y: update_procedure(x, y, master=master) if DoUpdate else None)
    edges = generate_edges(graph)
    weights = give_edge_weights(edges, result['best_of_all_gen'].dna)
    draw_graph(master, nx.DiGraph(graph), main_nodes_pos, edges, weights, f1_best_gen_graph, canvas=best_graph_canvas)
    best_info.config(text = f"""Balanced: {bool(sum(result['best_of_all_gen'].CheckBalanced()))}
Fitness score: {result['best_of_all_gen'].fitness_score: .4f}
Max Flow: {alg.col_sum(result['best_of_all_gen'].dna, -1)}
Generation: {result['gen_of_best_ind']}
Total Generation Ran: {result['total_gen']}""")
    
    enablef1butts()

def update_procedure(iter: int, best_individual: alg.Individual, *, master: tk.Tk) -> None:
    isbalanced = bool(sum(best_individual.CheckBalanced()))
    best_info.config(text=f"""Balanced: {isbalanced}
Fitness score: {best_individual.fitness_score: .4f}
Max Flow: {alg.col_sum(best_individual.dna, -1)}
Generation: {iter}""")
    edges = generate_edges(graph)
    weights = give_edge_weights(edges, best_individual.dna)
    draw_graph(master, nx.DiGraph(graph), main_nodes_pos, edges, weights, f1_best_gen_graph,
               canvas=best_graph_canvas)
    root.update()

def ToggleUpdateProd(button: tk.Button) -> None:
    global DoUpdate
    DoUpdate = not DoUpdate
    button.config(text=f"Toggle Update: {'On' if DoUpdate else 'Off'}",
                  bg='#007200' if DoUpdate else '#f0f0f0',
                  fg='white' if DoUpdate else 'black')

def SetPopSize(entryWidget: tk.Entry) -> None:
    error: bool = False
    input: str | list[str] = entryWidget.get().replace(' ', '')
    try:
        input = int(input)
        if input <= 0:
            raise ValueError()
    except ValueError:
        print("Input must be a positive integer...")
        error = True
    except TypeError:
        print("No input was given...")
        error = True
    if error:
        return
    
    global pop_size
    pop_size = input
    entryWidget.delete(0, 'end')
    UpdateF1Inf()
def SetMutRate(entryWidget: tk.Entry) -> None:
    error: bool = False
    input: str | list[str] = entryWidget.get().replace(' ', '')
    try:
        input = float(input)
        if input < 0 or input > 1:
            raise ValueError()
    except ValueError:
        print("Input must be a positive number between 0 and 1...")
        error = True
    except TypeError:
        print("No input was given...")
        error = True
    if error:
        return
    
    global mutation_rate
    mutation_rate = input
    entryWidget.delete(0, 'end')
    UpdateF1Inf()
def SetCrossFunc() -> None:
    global crossover_func
    crossover_func = crossover_func % 3 + 1
    UpdateF1Inf()
def SetMaxGen(entryWidget: tk.Entry) -> None:
    error: bool = False
    input: str | list[str] = entryWidget.get().replace(' ', '')
    try:
        input = int(input)
        if input <= 0:
            raise ValueError()
    except ValueError:
        print("Input must be a positive integer...")
        error = True
    except TypeError:
        print("No input was given...")
        error = True
    if error:
        return
    
    global max_iter
    max_iter = input
    entryWidget.delete(0, 'end')
    UpdateF1Inf()

def UpdateF1Inf() -> None:
    informations.config(text=f"""Crossover Function In Use: {crossover_func}
                        
Population Size: {pop_size}

Mutation Rate: {mutation_rate}

Max Generations: {max_iter}""")

def disablef1butts() -> None:
    instruc_btn['state'] = tk.DISABLED
    home['state'] = tk.DISABLED
    edit_graph['state'] = tk.DISABLED
    runFordFulkButt['state'] = tk.DISABLED
    runGenAlgButt['state'] = tk.DISABLED
    SetPopSizeButt['state'] = tk.DISABLED
    SetMutRateButt['state'] = tk.DISABLED
    SetMaxGenButt['state'] = tk.DISABLED
    SetCrossFuncButt['state'] = tk.DISABLED
    ToggleUpdateButt['state'] = tk.DISABLED
    PrematureStopButt['state'] = tk.DISABLED
    root.update()
def enablef1butts() -> None:
    instruc_btn['state'] = tk.NORMAL
    home['state'] = tk.NORMAL
    edit_graph['state'] = tk.NORMAL
    runFordFulkButt['state'] = tk.NORMAL
    runGenAlgButt['state'] = tk.NORMAL
    SetPopSizeButt['state'] = tk.NORMAL
    SetMutRateButt['state'] = tk.NORMAL
    SetMaxGenButt['state'] = tk.NORMAL
    SetCrossFuncButt['state'] = tk.NORMAL
    ToggleUpdateButt['state'] = tk.NORMAL
    PrematureStopButt['state'] = tk.NORMAL
    root.update()

def PrematureStopProd(button: tk.Button) -> None:
    global PrematureStop
    PrematureStop = not PrematureStop
    button.config(text=f"Premature Stop: {'On' if PrematureStop else 'Off'}",
                  bg='#007200' if PrematureStop else '#f0f0f0',
                  fg='white' if PrematureStop else 'black')
def _on_mousewheel(event):
   instruc_scroll.yview_scroll(int(-1 * (event.delta / 120)), "units")
# Define a grid
# Column
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.columnconfigure(2, weight=1)
# Row
root.rowconfigure(0, weight=0)   # menu
root.rowconfigure(1, weight=0)   # instruction
root.rowconfigure(2, weight=1)   # main frame
root.rowconfigure(3, weight=0)   # edit frame

# Create Buttons
instruc_btn = tk.Button(root, text='?', bg='white', font=('Arial', 13),
                        command=toggle_instruction_frame)
home = tk.Button(root, text='Home', bg='#064789', fg='white', font=('SVN-Gotham Bold', 13),
                    command=show_frame1)
edit_graph = tk.Button(root, text='Edit Graph', bg='#427aa1', fg='white', font=('SVN-Gotham Bold', 13),
                    command=show_frame2)

instruc_btn.grid(column=0, row=0, sticky=tk.NS,
                 padx=2, pady=2,
                 ipadx=10, ipady=1)
home.grid(row=0, column=1, sticky=tk.NS,
             padx=2, pady=2,
             ipadx=10, ipady=1)
edit_graph.grid(row=0, column=2, sticky=tk.NS, 
             padx=2, pady=2, 
             ipadx=10, ipady=1)

# Instruction (Button 1)
instruc_lb1 = tk.Frame(root)

instruc_lb1.grid(column=0, row=1, sticky='ew', columnspan=3,padx=20,pady=10)

instruc_scroll = tk.Canvas(instruc_lb1,width=1200,height=100,bg='light gray')
scrollbar = tk.Scrollbar(instruc_lb1, orient="vertical", command=instruc_scroll.yview)
instruc_scroll.configure(yscrollcommand=scrollbar.set)

instruc_content = tk.Frame(instruc_scroll)
instruc_content.bind("<Configure>", lambda e: instruc_scroll.configure(scrollregion=instruc_scroll.bbox("all")))

content = tk.Message(instruc_content,bg='light gray',fg='black',font=('SVN-Gotham Medium', 10),
                     text=f1_meta_information,width=1200)
content.grid(column=0, row=0, sticky='ew',columnspan=3)
instruc_scroll.create_window((0, 0), window=instruc_content, anchor="nw")
instruc_scroll.bind_all("<MouseWheel>", _on_mousewheel)
instruc_scroll.grid(row=0, column=0, sticky="nsew")
scrollbar.grid(row=0, column=3, sticky="ns")
instruc_lb1.grid_remove()
instruc_lb1.columnconfigure(0, weight=1)
instruc_lb1.rowconfigure(0, weight=1)


# Main Frame 1
main_frame1 = tk.Frame(root)
main_fm1 = tk.Label(main_frame1, bg="white",width=500,height=100)
main_frame1.grid(column=0, row=2, sticky='nsew', padx=20, pady=10, columnspan=3)
main_fm1.pack(fill='both', expand=True, ipadx=10, ipady=10)

# Frame 1's Widgets
null_flow = [[0] * len(capacity_matrix) for _ in range(len(capacity_matrix))]
edges = generate_edges(graph)
# Main Graph
main_graph = tk.Label(main_frame1, text="Main Graph", font=('SVN-Gotham Bold', 13),bg='white')
main_graph.place(relx=f1_main_graph[0] + graph_name_offset[0], rely=graph_name_offset[1])
main_graph_canvas = draw_graph(main_frame1, nx.DiGraph(graph), main_nodes_pos,
                               edges, give_edge_weights(edges, capacity_matrix),
                               f1_main_graph)
# Ford Fulkerson
ff_info = tk.Message(main_frame1, font=('SVN-Gotham Medium', 10), width=300,bg='white')
ff_graph = tk.Label(main_frame1, text="Ford Fulkerson", font=('SVN-Gotham Bold', 13),bg='white')
ff_graph.place(relx=f1_ff_graph[0] + graph_name_offset[0], rely=graph_name_offset[1])
ff_info.place(relx=f1_ff_graph[0] + info_box_offset[0], rely=f1_ff_graph[1] + info_box_offset[1])
ff_graph_canvas = draw_graph(main_frame1, nx.DiGraph(graph), main_nodes_pos,
                             edges, give_edge_weights(edges, null_flow),
                             f1_ff_graph)
# Best of all Gen
best_info = tk.Message(main_frame1, font=('SVN-Gotham Medium', 10), width=300,bg='white')
best_graph = tk.Label(main_frame1, text="Best Individual", font=('SVN-Gotham Bold', 13),bg='white')
best_info.place(relx=f1_best_gen_graph[0] + info_box_offset[0], rely=f1_best_gen_graph[1] + info_box_offset[1])
best_graph.place(relx=f1_best_gen_graph[0] + graph_name_offset[0], rely=graph_name_offset[1])
best_graph_canvas = draw_graph(main_frame1, nx.DiGraph(graph), main_nodes_pos,
                               edges, give_edge_weights(edges, null_flow),
                               f1_best_gen_graph)

informations = tk.Message(main_frame1, text=f1_information,
                          width=300, font=('SVN-Gotham Medium', 13),
                          bg='white')
informations.place(relx=0.77, rely=0.16)

# Edit Frame 1
edit_fm1 = tk.Frame(root, bg='lightgray')
edit_fm1.grid(column=0, row=3, sticky='ew', padx=20, pady=10, columnspan=3)
edit_fm1.grid_remove()
edit_lbl = tk.Label(edit_fm1, text='Edit:', bg='lightgray',font=('SVN-Gotham', 10))
edit_lbl.grid(row=0, column=0, padx=5, pady=5)
#Entry 1
entry1 = tk.Entry(edit_fm1, width=50)
entry1.grid(row=0, column=1, padx=5, pady=5)

# Button (Edit frame 1)
runFordFulkButt = tk.Button(edit_fm1, text="Ford Fulkerson", width=15,font=('SVN-Gotham Medium', 10),
                            command=lambda: RunFordFulkerson(main_frame1))
runGenAlgButt = tk.Button(edit_fm1, text="Genetic Algorithm", width=15,font=('SVN-Gotham Medium', 10),
                          command=lambda: RunGenAlg(main_frame1))
SetPopSizeButt = tk.Button(edit_fm1, text="Set Population Size", width=15, font=('SFT Futura Medium', 10),
                           command=lambda: SetPopSize(entry1))
SetMutRateButt = tk.Button(edit_fm1, text="Set Mutation Rate", width=15, font=('SFT Futura Medium', 10),
                           command=lambda: SetMutRate(entry1))
SetMaxGenButt = tk.Button(edit_fm1, text="Set Max Generation", width=22, font=('SFT Futura Medium', 10),
                           command=lambda: SetMaxGen(entry1))
SetCrossFuncButt = tk.Button(edit_fm1, text="Change Crossover Function", width=22, font=('SFT Futura Medium', 10),
                           command=lambda: SetCrossFunc())
ToggleUpdateButt = tk.Button(edit_fm1, text="Toggle Update: Off", width=15, font=('SFT Futura Medium', 10))
ToggleUpdateButt.config(command=lambda: ToggleUpdateProd(ToggleUpdateButt))
PrematureStopButt = tk.Button(edit_fm1, text="Premature Stop: On", width=15, font=('SFT Futura Medium', 10),
                              bg='#007200', fg='white')
PrematureStopButt.config(command=lambda: PrematureStopProd(PrematureStopButt))

runFordFulkButt.grid(row=0, column=2, padx=5, pady=5)
runGenAlgButt.grid(row=1, column=2, padx=5, pady=5)
SetPopSizeButt.grid(row=0, column=3,padx=5, pady=5)
SetMutRateButt.grid(row=1, column=3,padx=5, pady=5)
SetMaxGenButt.grid(row=0, column=4,padx=5, pady=5)
SetCrossFuncButt.grid(row=1, column=4,padx=5, pady=5)
ToggleUpdateButt.grid(row=0, column=5, padx=5, pady=5)
PrematureStopButt.grid(row=1, column=5, padx=5, pady=5)

# Main Frame 2
main_frame2 = tk.Frame(root)
main_frame2.grid(column=0, row=2, sticky='nsew', padx=20, pady=10, columnspan=3)
main_fm2 = tk.Label(main_frame2, bg="white", width=500, height=100)
main_fm2.pack(fill='both', expand=True, ipadx=10, ipady=10)

instruc_label = tk.Label(main_frame2,bg='light gray',fg='black',font=('SVN-Gotham Bold', 13), 
                        text = 'Instruction',padx=5,pady=5)
instruc_label.place(relx=0.25,rely=0.05,anchor='nw') # top left  

edit_graph_instruction = tk.Message(main_frame2,bg='#ebf7ff',font=('SVN-Gotham Medium', 10),
                        text=edit_instructions,width=1200)
edit_graph_instruction.place(relx=0.05,rely=0.15,anchor='nw') # top left
edit_graph_canvas = draw_graph(main_frame2, nx.DiGraph(graph), main_nodes_pos,
                               (temp := generate_edges(graph)), give_edge_weights(temp, capacity_matrix),
                               f2_graph)

# Edit Frame 2
edit_fm2 = tk.Frame(root, bg='lightgray')
edit_fm2.grid(column=0, row=3, sticky='ew', padx=20, pady=10, columnspan=3)
edit_fm2.grid_remove()
edit_lbl = tk.Label(edit_fm2, text='Edit:', bg='lightgray',font=('SVN-Gotham', 10))
edit_lbl.grid(row=0, column=0, padx=5, pady=5)

#Entry 2
entry2 = tk.Entry(edit_fm2, width=50)
entry2.grid(row=0, column=1, padx=5, pady=5)

#Button (Edit Frame 2)
addEdgeButt = tk.Button(edit_fm2, text="Add Edge",width=12,font=('SVN-Gotham Medium', 10),
                        command=lambda: AddEdgeInput(main_frame2, entry2))
addVertexButt = tk.Button(edit_fm2, text="Add Vertex",width=12,font=('SVN-Gotham Medium', 10),
                          command=lambda: AddVertexInput(main_frame2, entry2))
removeVertexButt = tk.Button(edit_fm2, text="Remove Vertex",width=12,font=('SVN-Gotham Medium', 10),
                            command=lambda: RemoveVertex(main_frame2, entry2))
removeEdgeButt = tk.Button(edit_fm2, text="Remove Edge",width=12,font=('SVN-Gotham Medium', 10),
                           command=lambda: RemoveEdge(main_frame2, entry2))
clearGraphButt = tk.Button(edit_fm2, text="Clear", width=12, font=('SVN Gotham Medium', 10), bg='#f4a261',fg='black',
                        command=lambda: ClearGraph(main_frame2))

addEdgeButt.grid(row=0, column=2, padx=5, pady=5)
addVertexButt.grid(row=1, column=2, padx=5, pady=5)
removeVertexButt.grid(row=1, column=3, padx=5, pady=5)
removeEdgeButt.grid(row=0, column=3, padx=5, pady=5)
clearGraphButt.grid(row=0, column=4, padx=5, pady=5)

show_frame1()

root.mainloop()
