#define GLM_FORCE_RADIANS

#include "common.h"
#include "sph.h"
#include "iisph/iisph.h"
#include "common/colored_output.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <cstdio>

#include <GL/glew.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>

#ifndef GLM_SWIZZLE
#define GLM_SWIZZLE 
#endif /* ifndef GLM_SWIZZLE */
#include <glm/gtc/type_ptr.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <sph_boundary_particles/ss.h>
#include <sph_boundary_particles/boundary_forces.h>

/**********************************************************************
 *                WINDOW AND MOUSE/KEYBOARD PARAMETERS                *
 **********************************************************************/
GLFWwindow* window;

int width = 1024;
int height = 768;

/**********************************************************************
*                EXPORT VIDEO LIKE IN MMACKLIN BLOG                  *
**********************************************************************/

const char* cmd = "ffmpeg -r 100 -f rawvideo -pix_fmt rgba -s 1024x768 -i - "
                  "-threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip output.mp4";

int* buffer = new int[width*height];
FILE* ffmpeg;

void openVideoStream()
{
	ffmpeg = popen(cmd, "w");
}

void exportFrame(GLFWwindow* win)
{
	glfwSwapBuffers(win);
	glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buffer);

	fwrite(buffer, sizeof(int)*width*height, 1, ffmpeg);
}

void closeVideoStream()
{
	pclose(ffmpeg);
}

/**********************************************************************
 *                           BASIC SHADERS                            *
 **********************************************************************/

const char* vertex_shader_spheres =
"#version 400\n"
"uniform float pointScale;"
"uniform float pointRadius;"
"uniform mat4 MVP;"
"uniform mat4 MV;"
"in vec4 vp;"
"in vec4 col;"
"out vec4 fcol;"
"void main() {"
"  vec3 posEye = vec3(MVP*vec4(vp));"
"  float dist = length(posEye);"
"  gl_PointSize = 0.017 * (pointScale/dist);"
"  gl_Position = MVP*vec4(vp);"
"  fcol = col;"
"}";

const char* fragment_shader_spheres =
"#version 400\n"
"const float PI = 3.1415926535897932384626433832795;"
"out vec4 frag_colour;"
"in vec4 fcol;"
"void main() "
"{"
"if(dot(gl_PointCoord-0.5,gl_PointCoord-0.5)>0.25) "
"discard;"
"else"
"{"
"vec3 lightDir = vec3(0.3,0.3,0.9);"
"vec3 N;"
"N.xy = gl_PointCoord* 2.0 - vec2(1.0);"
"float mag = dot(N.xy, N.xy);"
"N.z = sqrt(1.0-mag);"
"float diffuse = max(0.0, dot(lightDir, N));"
"frag_colour = vec4(fcol)*diffuse;"
"}"
"}";

const char * vertex_shader_basic =
"#version 400\n"
"uniform mat4 MVP;"
"in vec4 vp;"
"in vec4 col;"
"out vec4 fcol;"
"void main() {"
"  fcol = col;"
"  gl_Position = MVP*vp;"
"}";

const char * fragment_shader_basic = 
"#version 400\n"
"in vec4 fcol;"
"out vec4 frag_colour;"
"void main() {"
"  frag_colour = fcol;"
"}";

/**********************************************************************
 *                       SOME HELPER FUNCTIONS                        *
 **********************************************************************/

SReal particle_radius = 0.02;
SVec4 cube_points[8];SVec4 cube_colors[8]; unsigned int cube_indices[36];

void initCube()
{
	cube_points[0] = make_SVec4(-1,-1,2,1.f);
	cube_points[1] = make_SVec4(-1,-1,-1,1.f);
	cube_points[2] = make_SVec4(-1,2,-1,1.f);
	cube_points[3] = make_SVec4(-1,2,2,1.f);
	cube_points[4] = make_SVec4(2,-1,2,1.f);
	cube_points[5] = make_SVec4(2,-1,-1,1.f);
	cube_points[6] = make_SVec4(2,2,-1,1.f);
	cube_points[7] = make_SVec4(2,2,2,1.f);

	cube_colors[0] = make_SVec4(1,0,0,1);
	cube_colors[1] = make_SVec4(1,0,0,1);
	cube_colors[2] = make_SVec4(0,1,0,1);
	cube_colors[3] = make_SVec4(0,1,0,1);
	cube_colors[4] = make_SVec4(0,0,1,1);
	cube_colors[5] = make_SVec4(0,0,1,1);
	cube_colors[6] = make_SVec4(1,0,0,1);
	cube_colors[7] = make_SVec4(1,0,0,1);

	//front
	cube_indices[0] = 0; cube_indices[1] = 4; cube_indices[2] = 3;
	cube_indices[3] = 4; cube_indices[4] = 3; cube_indices[5] = 7;

	//back
	cube_indices[6] = 1; cube_indices[7] = 5; cube_indices[8] = 2;
	cube_indices[9] = 5; cube_indices[10] = 2; cube_indices[11] = 6;

	//left
	cube_indices[12] = 0; cube_indices[13] = 1; cube_indices[14] = 2;
	cube_indices[15] = 0; cube_indices[16] = 2; cube_indices[17] = 3;

	//right
	cube_indices[18] = 4; cube_indices[19] = 5; cube_indices[20] = 6;
	cube_indices[21] = 4; cube_indices[22] = 6; cube_indices[23] = 7;

	//top
	cube_indices[24] = 3; cube_indices[25] = 7; cube_indices[26] = 6;
	cube_indices[27] = 3; cube_indices[28] = 6; cube_indices[29] = 2;

	//bottom
	cube_indices[30] = 0; cube_indices[31] = 4; cube_indices[32] = 5;
	cube_indices[33] = 0; cube_indices[34] = 5; cube_indices[35] = 1;
}

void initWindow()
{
	if (!glfwInit())
	{
		fprintf(stderr, "ERROR: could not start GLFW3\n");
		exit(1);
	} 
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	window = glfwCreateWindow(width, height, "Hello Spheres", NULL, NULL);
	if (!window)
	{
		fprintf(stderr, "ERROR: could not open window with GLFW3\n");
		glfwTerminate();
		exit(1);
	}
	glfwMakeContextCurrent(window);

	// start GLEW extension handler
	glewExperimental = GL_TRUE;
	glewInit();

	// get version info
	const GLubyte* renderer = glGetString(GL_RENDERER); // get renderer string
	const GLubyte* version = glGetString(GL_VERSION); // version as a string
	printf("Renderer: %s\n", renderer);
	printf("OpenGL version supported %s\n", version);
}

void glEnableCapabilities()
{
	glEnable(GL_DEPTH_TEST); // enable depth-testing
	glDepthFunc(GL_LESS); // depth-testing interprets a smaller value as "closer"
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glEnable(GL_POINT_SPRITE);
	glEnable(GL_SMOOTH);
	glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
	glClearColor(1,1,1,1);
}

void getNewVbo(GLenum target, GLuint *newVbo, unsigned int bufferSize, const GLvoid* data, GLenum usage)
{
	glGenBuffers(1, newVbo);
	glBindBuffer(target, *newVbo);
	glBufferData(target, bufferSize, data, usage);
}

void getNewVao(GLuint *newVao, GLuint vbo_pos)
{
	glGenVertexArrays(1, newVao);
	glBindVertexArray(*newVao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
	glVertexAttribPointer(0, 4, GL_REAL, GL_FALSE, 0, NULL);
}

void getNewVao(GLuint *newVao, GLuint vbo_pos, GLuint vbo_col)
{
	glGenVertexArrays(1, newVao);
	glBindVertexArray(*newVao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
	glVertexAttribPointer(0, 4, GL_REAL, GL_FALSE, 0, NULL);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_col);
	glVertexAttribPointer(1, 4, GL_REAL, GL_FALSE, 0, NULL);
}

void getNewVao(GLuint *newVao, GLuint vbo_pos, GLuint vbo_col, GLuint vbo_indices)
{
	glGenVertexArrays(1, newVao);
	glBindVertexArray(*newVao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
	glVertexAttribPointer(0, 4, GL_REAL, GL_FALSE, 0, NULL);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_col);
	glVertexAttribPointer(1, 4, GL_REAL, GL_FALSE, 0, NULL);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices);
	glVertexAttribPointer(2, 4, GL_REAL, GL_FALSE, 0, NULL);
}

void compileVertexAndFragmentShaders(GLuint *vs, GLuint *fs, const GLchar **string_vs, const GLchar **string_fs)
{
	*vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(*vs, 1, string_vs, NULL);
	glCompileShader(*vs);
	*fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(*fs, 1, string_fs, NULL);
	glCompileShader(*fs);
}

void compileShaderProgram(GLuint *sp, GLuint vs, GLuint fs)
{
	*sp = glCreateProgram();
	glAttachShader(*sp, fs);
	glAttachShader(*sp, vs);
	glLinkProgram(*sp);
}

void displaySpheres(glm::mat4 mat_mvp, glm::mat4 mat_mv, GLuint shader_program, GLuint vao, GLuint vbo_pos, unsigned int nbSpheres)
{
	glUseProgram(shader_program);

	glBindVertexArray(vao);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	glUniform3f(glGetUniformLocation(shader_program, "col"), 0, 1, 0);
	glUniform1f(glGetUniformLocation(shader_program, "pointScale"), height / tanf(45.f*0.5f*(float)M_PI/180.0f));
	glUniform1f(glGetUniformLocation(shader_program, "pointRadius"), particle_radius);

	glUniformMatrix4fv(glGetUniformLocation(shader_program, "MVP"), 1, false, glm::value_ptr(mat_mvp));
	glUniformMatrix4fv(glGetUniformLocation(shader_program, "MV"), 1, false, glm::value_ptr(mat_mv));

	glDrawArrays(GL_POINTS, 0, nbSpheres); 
	glUseProgram(0);
}

void displayCube(glm::mat4 mat_mvp, GLuint shader_program, GLuint vao, GLuint vbo_pos, GLuint vbo_color, GLuint vbo_indices)
{
	glUseProgram(shader_program);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glBindVertexArray(vao);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glUniformMatrix4fv(glGetUniformLocation(shader_program, "MVP"), 1, false, glm::value_ptr(mat_mvp));

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices);

	glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, (void*)0); //TODO : replace 36 by a variable size
	glUseProgram(0);
}

/**********************************************************************
*                              GLOBALS                               *
**********************************************************************/

GLuint vbo_spheres_pos, vbo_spheres_col, vao_spheres;
GLuint vao_cube, vbo_cube_pos, vbo_cube_color, vbo_cube_indices;
GLuint vs_sphere, fs_sphere, vs_basic, fs_basic;

GLuint shader_program_spheres, shader_program_basic;

glm::mat4 mvp;
float fov = 45.f;
glm::vec4 campos(4,0,0,1);

bool do_simulation = false;

/**********************************************************************
 *                            KEY CALLBACK                            *
 **********************************************************************/

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
	{
        glfwSetWindowShouldClose(window, GLFW_TRUE);
		std::cout << "bye !" << std::endl;
	}

    if (key == GLFW_KEY_P && action == GLFW_PRESS)
	{
		if (do_simulation)
		{
			std::cout << "SIMULATION PAUSED" << std::endl;
		}
		else
		{
			std::cout << "SIMULATION RESUMED" << std::endl;
		}
		do_simulation = !do_simulation;
	}
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	fov += yoffset*0.02;
}

void move_camera_direction(GLFWwindow* win, glm::vec4* dir)
{
	int mouse_left_state = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT);
	if (mouse_left_state == GLFW_PRESS || mouse_left_state == GLFW_REPEAT)
	{
		double xcenter = width / 2.;
		double ycenter = height / 2.;
		double xpos, ypos;
		glfwGetCursorPos(win, &xpos, &ypos);

		if (xpos < xcenter)
		{
			dir->x -= 0.01f;
		}

		if (xpos > xcenter)
		{
			dir->x += 0.01f;
		}

		if (ypos < ycenter)
		{
			dir->y += 0.01f;
		}

		if (ypos > ycenter)
		{
			dir->y -= 0.01f;
		}

		glfwSetCursorPos(win, width/2, height/2);
	}
}

void move_camera_rotate(GLFWwindow * win, glm::mat4 *mvp)
{
	static float rotateAroundY = 0.f;
	static float rotateAroundX = 0.f;

	int mouse_left_state = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT);
	if (mouse_left_state == GLFW_PRESS || mouse_left_state == GLFW_REPEAT)
	{
		double xcenter = width / 2.;
		double ycenter = height / 2.;
		double xpos, ypos;
		glfwGetCursorPos(win, &xpos, &ypos);


		if (xpos < xcenter)
		{
			rotateAroundY -= 0.005f;
		}

		if (xpos > xcenter)
		{
			rotateAroundY += 0.005f;
		}

		if (ypos < ycenter)
		{
			rotateAroundX -= 0.005f;
		}

		if (ypos > ycenter)
		{
			rotateAroundX += 0.005f;
		}
		 

		glfwSetCursorPos(win, width/2, height/2);
	}
	*mvp = glm::rotate(*mvp, rotateAroundY, glm::vec3(0,1,0));
	*mvp = glm::rotate(*mvp, rotateAroundX, glm::vec3(1,0,0));
}

glm::vec4 getCamMove(GLFWwindow *win, glm::vec4 camDir, glm::vec4 camUp)
{
	static glm::vec4 res(0,0,0,0);
	
	glm::vec4 right = glm::vec4(glm::cross(camUp.xyz(), camDir.xyz()),0);

	int stateA = glfwGetKey(win, GLFW_KEY_A);
	if (stateA == GLFW_PRESS || stateA == GLFW_REPEAT)
	{
		res -= 0.001f * right;
	}

	int stateD = glfwGetKey(win, GLFW_KEY_D);
	if (stateD == GLFW_PRESS || stateD == GLFW_REPEAT)
	{
		res += 0.001f * right;
	}

	int stateW = glfwGetKey(win, GLFW_KEY_W);
	if (stateW == GLFW_PRESS || stateW == GLFW_REPEAT)
	{
		res -= 0.001f * camDir;
	}

	int stateS = glfwGetKey(win, GLFW_KEY_S);
	if (stateS == GLFW_PRESS || stateS == GLFW_REPEAT)
	{
		res += 0.001f * camDir;
	}

	int stateSp = glfwGetKey(win, GLFW_KEY_SPACE);
	if (stateSp == GLFW_PRESS || stateSp == GLFW_REPEAT)
	{
		res += glm::vec4(0, 0.001f, 0, 0);
	}

	int stateC = glfwGetKey(win, GLFW_KEY_C);
	if (stateC == GLFW_PRESS || stateSp == GLFW_REPEAT)
	{
		res += glm::vec4(0, -0.001f, 0, 0);
	}


	return res;
}

/**********************************************************************
*                            FPS COUNTER                             *
**********************************************************************/
double lastTime = time(NULL);
void displayFPS(GLFWwindow *window)
{
	static int nbFrame = 0;
	time_t currentTime;
	time(&currentTime);

	nbFrame++;
	if (currentTime - lastTime >= 1.0)
	{
		std::ostringstream ss;
		ss << "SPH Simulator - FPS = " << (float)nbFrame;
		//std::cout << "fps = " << (float)nbFrame << std::endl;	
		glfwSetWindowTitle(window, ss.str().c_str());
		nbFrame = 0;
		lastTime = currentTime;
	}
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void drop_more_particles(GLFWwindow* win, CFD::SPH *sim_sph)
{
	static bool dropped = false;
	int stateK = glfwGetKey(win, GLFW_KEY_K);
	if (stateK == GLFW_PRESS && dropped == false)
	{
		dropped = true;
		sim_sph->generateParticleCube(make_SVec4(-0.f, 1.4f, 0.0f,1.f), make_SVec4(0.5f, 0.5f, 0.5f, 0.f), make_SVec4(0,0,0,0));

		getNewVbo(GL_ARRAY_BUFFER, &vbo_spheres_pos, sim_sph->getNumParticles() * sizeof(SVec4), sim_sph->getHostPos(), GL_STATIC_DRAW);
		getNewVbo(GL_ARRAY_BUFFER, &vbo_spheres_col, sim_sph->getNumParticles() * sizeof(SVec4), sim_sph->getHostCol(), GL_STATIC_DRAW);
		getNewVao(&vao_spheres, vbo_spheres_pos, vbo_spheres_col);

		std::cout << "more particles" << std::endl;
	}

	//lock particle dropping to avoid explosions
	int stateU = glfwGetKey(win, GLFW_KEY_U);
	if(stateU == GLFW_PRESS && dropped == true)
	{
		dropped = false;
	}
}

/**********************************************************************
 *                            MAIN PROGRAM                            *
 **********************************************************************/
int main(void)
{
#if RECORD_SIMULATION == 1
	static unsigned int save = 0;
	openVideoStream();
#endif

	CFD::SPH *sim_sph = new CFD::SPH();
	sim_sph->_initialize();
	sim_sph->generateParticleCube(make_SVec4(-0.4f, 0.04f, 0.5f, 1.f), make_SVec4(1.0f, 2.0f, 2.9f, 1.f), make_SVec4(0,0,0,0));
	//sim_sph->generateParticleCube(make_SVec4(-0.4f, 0.04f, 0.5f, 1.f), make_SVec4(0.5f, 0.5f, 0.5f, 1.f), make_SVec4(0,0,0,0));

	//make boundary particles
	std::vector<SVec4> bi;
	std::vector<SReal> vbi;

	//FIXME VARIABLE FOR RADIUS
	sample_spheres::ss::sampleBox(bi, make_SVec3(-1, -1, -1), make_SVec3(3.f, 3.f, 3.f), 0.02 );
	sample_spheres::boundary_forces::getVbi(vbi, bi, sim_sph->getInteractionRadius());

	sim_sph->setNumBoundaries(bi.size());

	sim_sph->setBi((SReal*)bi.data());
	sim_sph->setVbi(vbi.data());

	sim_sph->updateGpuBoundaries(bi.size());

	std::cout << "boundary particles number " << bi.size() << std::endl;
	
	//call to helper functions
	initWindow();
	glfwSetKeyCallback(window, key_callback);
	glEnableCapabilities();
	initCube();

	//opengl sphere buffers handling
	getNewVbo(GL_ARRAY_BUFFER, &vbo_spheres_pos, sim_sph->getNumParticles() * sizeof(SVec4), sim_sph->getHostPos(), GL_STATIC_DRAW);
	getNewVbo(GL_ARRAY_BUFFER, &vbo_spheres_col, sim_sph->getNumParticles() * sizeof(SVec4), sim_sph->getHostCol(), GL_STATIC_DRAW);
	getNewVao(&vao_spheres, vbo_spheres_pos, vbo_spheres_col);

	//opengl cube buffers handling
	getNewVbo(GL_ARRAY_BUFFER, &vbo_cube_pos, 8 * sizeof(SVec4), cube_points, GL_STATIC_DRAW);
	getNewVbo(GL_ARRAY_BUFFER, &vbo_cube_color, 8 *sizeof(SVec4), cube_colors, GL_STATIC_DRAW);
	getNewVbo(GL_ELEMENT_ARRAY_BUFFER, &vbo_cube_indices, 36 * sizeof(unsigned int), cube_indices, GL_STATIC_DRAW);
	getNewVao(&vao_cube, vbo_cube_pos, vbo_cube_color, vbo_cube_indices);

	//shaders handling
	compileVertexAndFragmentShaders(&vs_sphere, &fs_sphere, &vertex_shader_spheres, &fragment_shader_spheres);
	compileShaderProgram(&shader_program_spheres, vs_sphere, fs_sphere);
	compileVertexAndFragmentShaders(&vs_basic, &fs_basic, &vertex_shader_basic, &fragment_shader_basic);
	compileShaderProgram(&shader_program_basic, vs_basic, fs_basic);

	glm::vec4 direction(0,0,1,0);

	while(!glfwWindowShouldClose(window))
	{
		drop_more_particles(window, sim_sph);
		displayFPS(window);

		glBindBuffer(GL_ARRAY_BUFFER, vbo_spheres_pos);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sim_sph->getNumParticles() * sizeof(SVec4), sim_sph->getHostPos());

		//step 1 : clear screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		move_camera_direction(window, &direction);

		//step 2 : handle mvp matrix
		glm::mat4 m(1.f);
		glm::vec4 camMove = getCamMove(window, direction, glm::vec4(0,1,0,0));
		glm::vec4 camPos = glm::vec4(0,0,4,1) + camMove;

		glm::mat4 v = glm::lookAt(camPos.xyz(), direction.xyz(), glm::vec4(0,1,0, 0).xyz());
		glm::mat4 p = glm::perspective(fov,(float)width/float(height), 0.1f, 100.f);
		mvp = p*v*m;
		move_camera_rotate(window,&mvp);

		glm::mat4 mv = v*m;

		//step 3 : display spheres in associated shader program
		displaySpheres(mvp, mv, shader_program_spheres, vao_spheres, vbo_spheres_pos, sim_sph->getNumParticles());

		//step 4 : display cube in associated shader program
		displayCube(mvp, shader_program_basic, vao_cube, vbo_cube_pos, vbo_cube_color, vbo_cube_indices);

		//step 5 : sph computations
		if(do_simulation)
		{
			sim_sph->update();

#if RECORD_SIMULATION == 1
			if (save % 10 == 0) 
			{
				exportFrame(window);
				save = 0;
			}
#endif
		}

		//last step : read new events if some
		glfwPollEvents();
		glfwSwapBuffers(window);

#if RECORD_SIMULATION == 1
		save++;
#endif
	}

	sim_sph->_finalize();

#if RECORD_SIMULATION == 1
	closeVideoStream();
#endif
	glfwTerminate();
	exit(EXIT_SUCCESS);
}
