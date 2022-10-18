import os
import subprocess
import weakref

def _dot_var(v, verbose=False): # variable 인스턴스의 내용을 DOT 언어로 작성된 문자열로 바꿔서 반환
    dot_var = '{} [label = "{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name =+ str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)

def _dot_func(f): # Dezero 함수를 DOT 언어로 기술
    dot_func = '{} [label="{}",color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))
    return txt

def get_dot_graph(output, verbose=True): # 계산 그래프 시각화 코드
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return 'digraph g {\n' + txt + '}'

def plot_dot_graph(output, verbose=True, to_file='graph.png'): # 계산 그래프의 DOT 언어를 이미지로 변환하는 명령어
    dot_graph = get_dot_graph(output, verbose)

    tmp_dir = os.path.join(os.getcwd(), 'graph_image') # 경로 생성

    if not os.path.exists(tmp_dir): # 위 경로에 해당하는 디렉터리가 없다면
        os.mkdir('graph_image') # 하나 만들어 준다

    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:] # 파일 확장명인 png만 가져온다.
    cmd = 'dot {} -T {} -o {}'.format(graph_path,extension,to_file)
    subprocess.run(cmd,shell=True)

