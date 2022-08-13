import ctypes
import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
import numpy as np

null = ctypes.c_void_p(0)

''' render vertex attributes, e.g., colors or normals '''
vs_vertex_attribute = '''
    #version 330 core
    uniform mat4 mvp;
    layout (location = 0) in vec3 vertices;
    layout (location = 1) in vec3 attributes;
    out vec4 vertex_attributes;
    void main(){
        gl_Position = mvp * vec4(vertices, 1.f);
        vertex_attributes = vec4(attributes, 1.f);
    }
    '''

fs_vertex_attribute = '''
    #version 330 core
    in vec4 vertex_attributes;
    out vec4 frag_color;
    void main(){
        frag_color = vertex_attributes;
    }
    '''

''' render vertex positions '''
vs_position = '''
    #version 330 core
    uniform mat4 mvp;
    layout (location = 0) in vec3 vertices;
    out vec4 positions;
    void main(){
        gl_Position = mvp * vec4(vertices, 1.f);
        positions = vec4(vertices, 1.f);
    }
    '''

fs_position = '''
    #version 330 core
    in vec4 positions;
    out vec4 frag_color;
    void main(){
        frag_color = positions;
    }
    '''

''' Render phong geometry '''
vs_phong_geometry = '''
    #version 330 core
    uniform mat4 mvp;
    uniform mat4 mv;
    layout (location = 0) in vec3 vertices;
    layout (location = 1) in vec3 normals;
    out VS_OUT
    {
        vec3 v;
        vec3 fn;
        vec3 bn;
    } vs_out;
    void main(){
        gl_Position = mvp * vec4(vertices, 1.f);
        
        vec4 v_cam = mv * vec4(vertices, 1.f);
        vs_out.v = v_cam.xyz;
        
        mat3 R = mat3(mv);
        vec3 front_normal = normalize(R * normals.xyz);
        vec3 back_normal = -front_normal;
        vs_out.fn = front_normal;
        vs_out.bn = back_normal;
    }
    '''

fs_phong_geometry = '''
    #version 330 core
    struct LightAttrib
    {
        vec3 la;
        vec3 ld;
        vec3 ls;
        vec3 ldir;
    };
    
    struct MaterialAttrib
    {
        vec3 ma;
        vec3 md;
        vec3 ms;
        float ss;
    };
    
    in VS_OUT
    {
        vec3 v;
        vec3 fn;
        vec3 bn;
    } fs_in;

    out vec4 frag_color;
    void main(){
        /* init lighting, front material and back material */
        LightAttrib light = LightAttrib(
            vec3(0.3, 0.3, 0.3),
            vec3(0.7, 0.7, 0.7),
            vec3(1.0, 1.0, 1.0),
            vec3(0.0, 0.0, 1.0)
        );
        
        // gray
        MaterialAttrib fmat = MaterialAttrib(
            vec3(0.85f, 0.85f, 0.85f),
            vec3(0.85f, 0.85f, 0.85f),
            vec3(0.1, 0.1, 0.1),
            10.0
        );

        MaterialAttrib bmat = MaterialAttrib(
            vec3(0.85, 0.85, 0.85),
            vec3(0.85, 0.85, 0.85),
            vec3(0.6, 0.6, 0.6),
            100.0
        );
        
        /*Calculate light, view, front-facing and back-facing normals*/
        vec3 ldir = normalize(light.ldir);
        vec3 fn = normalize(fs_in.fn);
        vec3 bn = normalize(fs_in.bn);
        vec3 vdir = normalize(-fs_in.v);
        vec3 frdir = reflect(-ldir, fn);
        vec3 brdir = reflect(-ldir, bn);
        
        /*discard this fragment if normal is NAN*/
        if (any(isnan(fn)) || any(isnan(bn))) discard;

        /*render double faces*/
        if (gl_FrontFacing) {
            /*calculate radiance*/
            vec3 ka = light.la * fmat.ma;
            vec3 kd = light.ld * fmat.md;
            vec3 ks = light.ls * fmat.ms;
            
            /*calculate Phong lighting of front-facing fragment*/
            vec3 fca = ka;
            vec3 fcd = kd * max(dot(fn, ldir), 0.0);
            vec3 fcs = ks * pow(max(dot(vdir, frdir), 0.0), fmat.ss);
            
            vec3 fc = clamp(fca + fcd + fcs, 0.0, 1.0);
            frag_color = vec4(fc, 1.0);
        }
        else {
            /*calculate radiance*/
            vec3 ka = light.la * bmat.ma;
            vec3 kd = light.ld * bmat.md;
            vec3 ks = light.ls * bmat.ms;
            
            /*calculate Phong lighting of back-facing fragment*/
            vec3 bca = ka;
            vec3 bcd = kd * max(dot(bn, ldir), 0.0);
            vec3 bcs = ks * pow(max(dot(vdir, brdir), 0.0), bmat.ss);
            
            vec3 bc = clamp(bca + bcd + bcs, 0.0, 1.0);
            frag_color = vec4(bc, 1.0);
        }
    }
    '''

''' Render phong color '''
vs_phong_color = '''
    #version 330 core
    uniform mat4 mvp;
    uniform mat4 mv;
    layout (location = 0) in vec3 vertices;
    layout (location = 1) in vec3 normals;
    layout (location = 2) in vec3 colors;
    out VS_OUT
    {
        vec3 v;
        vec3 fn;
        vec3 bn;
        vec3 c;
    } vs_out;
    void main(){
        gl_Position = mvp * vec4(vertices, 1.f);

        vec4 v_cam = mv * vec4(vertices, 1.f);
        vs_out.v = v_cam.xyz;

        mat3 R = mat3(mv);
        vec3 front_normal = normalize(R * normals.xyz);
        vec3 back_normal = -front_normal;
        vs_out.fn = front_normal;
        vs_out.bn = back_normal;
        vs_out.c = colors;
    }
    '''

fs_phong_color = '''
    #version 330 core
    struct LightAttrib
    {
        vec3 la;
        vec3 ld;
        vec3 ls;
        vec3 ldir;
    };

    struct MaterialAttrib
    {
        vec3 ma;
        vec3 md;
        vec3 ms;
        float ss;
    };

    in VS_OUT
    {
        vec3 v;
        vec3 fn;
        vec3 bn;
        vec3 c;
    } fs_in;

    out vec4 frag_color;
    void main(){
        /* init lighting, front material and back material */
        LightAttrib light = LightAttrib(
            vec3(0.3, 0.3, 0.3),
            vec3(0.7, 0.7, 0.7),
            vec3(1.0, 1.0, 1.0),
            vec3(0.0, 0.0, 1.0)
        );

        // gray
        MaterialAttrib fmat = MaterialAttrib(
            vec3(0.85f, 0.85f, 0.85f),
            vec3(0.85f, 0.85f, 0.85f),
            vec3(0.1, 0.1, 0.1),
            10.0
        );

        MaterialAttrib bmat = MaterialAttrib(
            vec3(0.85, 0.85, 0.85),
            vec3(0.85, 0.85, 0.85),
            vec3(0.6, 0.6, 0.6),
            100.0
        );

        /*Calculate light, view, front-facing and back-facing normals*/
        vec3 ldir = normalize(light.ldir);
        vec3 fn = normalize(fs_in.fn);
        vec3 bn = normalize(fs_in.bn);
        vec3 vdir = normalize(-fs_in.v);
        vec3 frdir = reflect(-ldir, fn);
        vec3 brdir = reflect(-ldir, bn);

        /*discard this fragment if normal is NAN*/
        if (any(isnan(fn)) || any(isnan(bn))) discard;

        /*render double faces*/
        if (gl_FrontFacing) {
            /*calculate radiance*/
            vec3 ka = light.la * fmat.ma;
            vec3 kd = light.ld * fmat.md;
            vec3 ks = light.ls * fmat.ms;

            /*calculate Phong lighting of front-facing fragment*/
            vec3 fca = ka;
            vec3 fcd = kd * max(dot(fn, ldir), 0.0);
            vec3 fcs = ks * pow(max(dot(vdir, frdir), 0.0), fmat.ss);

            vec3 fc = clamp(fca + fcd + fcs, 0.0, 1.0);
            frag_color = vec4(fc, 1.0) * vec4(fs_in.c, 1.0);
        }
        else {
            /*calculate radiance*/
            vec3 ka = light.la * bmat.ma;
            vec3 kd = light.ld * bmat.md;
            vec3 ks = light.ls * bmat.ms;

            /*calculate Phong lighting of back-facing fragment*/
            vec3 bca = ka;
            vec3 bcd = kd * max(dot(bn, ldir), 0.0);
            vec3 bcs = ks * pow(max(dot(vdir, brdir), 0.0), bmat.ss);

            vec3 bc = clamp(bca + bcd + bcs, 0.0, 1.0);
            frag_color = vec4(bc, 1.0) * vec4(fs_in.c, 1.0);
        }
    }
    '''


# Note that the model is in the usual camera space by default (not opengl)
def gl_perspective_projection_matrix(fx, fy, cx, cy, img_w, img_h, far = 100.0, near = 0.1, gl_space = False):
    proj_mat = np.zeros((4, 4), dtype = np.float32)
    proj_mat[0, 0] = 2 * fx / img_w
    proj_mat[0, 2] = (2 * cx - img_w) / img_w
    proj_mat[1, 1] = -2 * fy / img_h
    proj_mat[1, 2] = (img_h - 2 * cy) / img_h
    proj_mat[2, 2] = (far + near) / (far - near)
    proj_mat[2, 3] = 2 * near * far / (near - far)
    proj_mat[3, 2] = 1.
    if gl_space:
        real2gl = np.identity(4, dtype = np.float32)
        real2gl[1, 1] = -1
        real2gl[2, 2] = -1
        proj_mat = np.dot(proj_mat, real2gl)
    return proj_mat


# Note that the model is in the opengl camera space
def gl_orthographic_projection_matrix(far = -100.0, near = -0.1):
    proj_mat = np.zeros((4, 4), dtype = np.float32)
    proj_mat[0, 0] = 1.
    proj_mat[1, 1] = 1.
    proj_mat[2, 2] = 2 / (far - near)
    proj_mat[2, 3] = -(far + near) / (far - near)
    proj_mat[3, 3] = 1.
    return proj_mat


class Renderer:
    def __init__(self, img_w: int, img_h: int, mvp: np.ndarray = np.array(4, dtype = np.float32), shader_name = 'vertex_attribute', bg_color = (0, 0, 0), window_name = ''):
        glfw.init()
        self.img_w = img_w
        self.img_h = img_h
        self.bg_color = bg_color
        self.window = glfw.create_window(img_w, img_h, window_name, None, None)
        glfw.set_window_pos(self.window, 500, 500)
        glfw.hide_window(self.window)
        glfw.make_context_current(self.window)

        # init shader
        if shader_name == 'vertex_attribute':
            vertex_shader = shaders.compileShader(vs_vertex_attribute, GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(fs_vertex_attribute, GL_FRAGMENT_SHADER)
        elif shader_name == 'position':
            vertex_shader = shaders.compileShader(vs_position, GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(fs_position, GL_FRAGMENT_SHADER)
        elif shader_name == 'phong_geometry':
            vertex_shader = shaders.compileShader(vs_phong_geometry, GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(fs_phong_geometry, GL_FRAGMENT_SHADER)
        elif shader_name == 'phong_color':
            vertex_shader = shaders.compileShader(vs_phong_color, GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(fs_phong_color, GL_FRAGMENT_SHADER)
        else:
            raise ValueError('Invalid shader name!')
        self.shader = shaders.compileProgram(vertex_shader, fragment_shader)
        glUseProgram(self.shader)

        # init model, view, projection matrix
        self.uniform_locations = {'mvp': glGetUniformLocation(self.shader, 'mvp')}
        glUniformMatrix4fv(self.uniform_locations['mvp'], 1, GL_TRUE, mvp)

        # vertex and corresponding attribute location
        # self.attribute_locations = {'vertices': glGetAttribLocation(self.shader, 'vertices'),
        #                             'attributes': glGetAttribLocation(self.shader, 'attributes')}
        self.vao = glGenVertexArrays(1)
        self.vertices_vbo = glGenBuffers(1)
        self.attributes_vbo = glGenBuffers(1)
        self.attributes_vbo_2 = glGenBuffers(1)

        ''' frame buffer object '''
        # frame buffer
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        # texture
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, img_w, img_h, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

        # render buffer
        rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, img_w, img_h)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print('frame buffer is not complete!')
        glBindFramebuffer(GL_FRAMEBUFFER, 0)  # switch to default frame buffer

    def set_mvp_mat(self, mvp: np.ndarray):
        glfw.make_context_current(self.window)
        glUseProgram(self.shader)
        glUniformMatrix4fv(self.uniform_locations['mvp'], 1, GL_TRUE, mvp)

    def set_mv_mat(self, mv: np.ndarray):
        glfw.make_context_current(self.window)
        glUseProgram(self.shader)
        mv_location = glGetUniformLocation(self.shader, 'mv')
        glUniformMatrix4fv(mv_location, 1, GL_TRUE, mv)

    def set_model(self, vertices, vertex_attributes = None, vertex_attributes_2 = None):
        """
        the order of vertex attributes:
        1. normal
        2. color
        """
        glfw.make_context_current(self.window)
        glBindVertexArray(self.vao)
        # vertices
        glBindBuffer(GL_ARRAY_BUFFER, self.vertices_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.size * ctypes.sizeof(GLfloat), vertices, GL_STREAM_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, null)

        if vertex_attributes is not None:
            # attributes
            glBindBuffer(GL_ARRAY_BUFFER, self.attributes_vbo)
            glBufferData(GL_ARRAY_BUFFER, vertex_attributes.size * ctypes.sizeof(GLfloat), vertex_attributes, GL_STREAM_DRAW)
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 3, GL_FLOAT, False, 0, null)

        if vertex_attributes_2 is not None:
            # attributes
            glBindBuffer(GL_ARRAY_BUFFER, self.attributes_vbo_2)
            glBufferData(GL_ARRAY_BUFFER, vertex_attributes_2.size * ctypes.sizeof(GLfloat), vertex_attributes_2, GL_STREAM_DRAW)
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(2, 3, GL_FLOAT, False, 0, null)

        glBindVertexArray(0)

        self.vnum = vertices.shape[0]

    def render(self):
        glfw.make_context_current(self.window)
        glfw.poll_events()

        # defined frame buffer object
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glClearColor(self.bg_color[0], self.bg_color[1], self.bg_color[2], 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glEnable(GL_CULL_FACE)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vnum)
        data = glReadPixels(0, 0, self.img_w, self.img_h, GL_RGBA, GL_FLOAT)
        data = np.frombuffer(data, np.float32)
        data.shape = self.img_h, self.img_w, 4
        data = data[::-1, :]
        glBindVertexArray(0)
        return data
