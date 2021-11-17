import {defs, tiny} from './examples/common.js';

const {
    Vector, Vector3, vec, vec3, vec4, color, hex_color, Shader, Matrix, Mat4, Light, Shape, Material, Scene,
} = tiny;

export class Demo3 extends Scene {
    constructor() {
        // constructor(): Scenes begin by populating initial values like the Shapes and Materials they'll need.
        super();

        // At the beginning of our program, load one of each of these shape definitions onto the GPU.
        this.shapes = {
            torus: new defs.Torus(15, 15),
            torus2: new defs.Torus(3, 15),
            sphere: new defs.Subdivision_Sphere(4),
            circle: new defs.Regular_2D_Polygon(1, 15),
            // TODO:  Fill in as many additional shape instances as needed in this key/value table.
            //        (Requirement 1)
            s1: new defs.Subdivision_Sphere(1),
            s2: new defs.Subdivision_Sphere(2),
            s3: new defs.Subdivision_Sphere(3),
            s4: new defs.Subdivision_Sphere(4),
            f1: new (defs.Subdivision_Sphere.prototype.make_flat_shaded_version())(1),
            f2: new (defs.Subdivision_Sphere.prototype.make_flat_shaded_version())(2),
            f3: new (defs.Subdivision_Sphere.prototype.make_flat_shaded_version())(3),
            f4: new (defs.Subdivision_Sphere.prototype.make_flat_shaded_version())(4),
        };

        // *** Materials
        this.materials = {
            test: new Material(new defs.Phong_Shader(),
                {ambient: .4, diffusivity: .6, color: hex_color("#ffffff")}),
            ver1: new Material(new Shader_Version_1(), { color: hex_color("#00ffff") }),
            ver2: new Material(new Shader_Version_2(), { color: hex_color("#00ffff"), second_color: hex_color("#ff0000") }),
            ver3: new Material(new Shader_Version_3(), { color: hex_color("#00ffff") }),
            exp: new Material(new Example_Shader(), {})
        }

        this.initial_camera_location = Mat4.look_at(vec3(0, 5, 5), vec3(0, 0, 0), vec3(0, 1, 0));
    }

    make_control_panel() {
        // Draw the scene's buttons, setup their actions and keyboard shortcuts, and monitor live measurements.
        this.key_triggered_button("View solar system", ["Control", "0"], () => this.attached = () => null);
        this.new_line();
        this.key_triggered_button("Attach to planet 1", ["Control", "1"], () => this.attached = () => this.planet_1);
        this.key_triggered_button("Attach to planet 2", ["Control", "2"], () => this.attached = () => this.planet_2);
        this.new_line();
        this.key_triggered_button("Attach to planet 3", ["Control", "3"], () => this.attached = () => this.planet_3);
        this.key_triggered_button("Attach to planet 4", ["Control", "4"], () => this.attached = () => this.planet_4);
        this.new_line();
        this.key_triggered_button("Attach to moon", ["Control", "m"], () => this.attached = () => this.moon);
    }

    display(context, program_state) {
        // display():  Called once per frame of animation.
        let desired;
// Setup -- This part sets up the scene's overall camera matrix, projection matrix, and lights:
        if (!context.scratchpad.controls) {
            this.children.push(context.scratchpad.controls = new defs.Movement_Controls());
            // Define the global camera and projection matrices, which are stored in program_state.
            program_state.set_camera(this.initial_camera_location);
        }

        program_state.projection_transform = Mat4.perspective(
            Math.PI / 4, context.width / context.height, .1, 1000);

        const t = program_state.animation_time / 1000, dt = program_state.animation_delta_time / 1000;

        const light_position = vec4(0, 5, 5, 1);

        // The parameters of the Light are: position, color, size
        program_state.lights = [new Light(light_position, color(1, 1, 1, 1), 1000)];
        // // Use following for Point Light Source with Changing Size
        // program_state.lights = [new Light(light_position, color(1, 1, 1, 1), 10 ** (t % 3))];

        // // Shader Example 1
        // this.shapes.torus.draw(context, program_state, Mat4.scale(1, 1, 0.2), this.materials.ver1);

        // // Shader Example 2
        // this.shapes.torus.draw(context, program_state, Mat4.scale(1, 1, 0.2), this.materials.ver2);

        // // Shader Example 3
        // let model_transformation = Mat4.translation(3 * Math.sin(t), 0, 0).times(
        //     Mat4.scale(1, 1, 0.2)
        // );
        // this.shapes.torus.draw(context, program_state, model_transformation, this.materials.ver2);

        // // Shader Example 4
        // let model_transformation = Mat4.translation(3 * Math.sin(t), 0, 0.5).times(
        //     Mat4.scale(1, 1, 0.2)
        // );
        // this.shapes.torus.draw(context, program_state, model_transformation, this.materials.ver3);

        // // Subdivisions
        // this.shapes.s1.draw(context, program_state, Mat4.translation(-3, 0, 0), this.materials.test);
        // this.shapes.s2.draw(context, program_state, Mat4.translation(-1, 0, 0), this.materials.test);
        // this.shapes.s3.draw(context, program_state, Mat4.translation(+1, 0, 0), this.materials.test);
        // this.shapes.s4.draw(context, program_state, Mat4.translation(+3, 0, 0), this.materials.test);
        // // Flat Shapes
        // this.shapes.f1.draw(context, program_state, Mat4.translation(-3, 0, 0), this.materials.test);
        // this.shapes.f2.draw(context, program_state, Mat4.translation(-1, 0, 0), this.materials.test);
        // this.shapes.f3.draw(context, program_state, Mat4.translation(+1, 0, 0), this.materials.test);
        // this.shapes.f4.draw(context, program_state, Mat4.translation(+3, 0, 0), this.materials.test);

        // // Example for Requirement about the camera.
        // this.planet_1 = Mat4.translation(-3, 0, 0);
        // this.planet_2 = Mat4.translation(-1, 0, 0);
        // this.planet_3 = Mat4.translation(+1, 0, 0);
        // this.planet_4 = Mat4.translation(+3, 0, 0);
        //
        // if (this.attached && this.attached() !== null) {
        //     desired = Mat4.inverse(this.attached().times(Mat4.translation(0, 0, 10)));
        // } else {
        //     desired = this.initial_camera_location;
        // }
        // program_state.set_camera(desired);

        // Example Shader for Week6
        this.shapes.s1.draw(context, program_state, Mat4.translation(-3, 0, -2), this.materials.exp);
        this.shapes.s2.draw(context, program_state, Mat4.translation(-1, 0, -2), this.materials.exp);
        this.shapes.s3.draw(context, program_state, Mat4.translation(+1, 0, -2), this.materials.exp);
        this.shapes.s4.draw(context, program_state, Mat4.translation(+3, 0, -2), this.materials.exp);
        this.shapes.f1.draw(context, program_state, Mat4.translation(-3, 0, 0), this.materials.exp);
        this.shapes.f2.draw(context, program_state, Mat4.translation(-1, 0, 0), this.materials.exp);
        this.shapes.f3.draw(context, program_state, Mat4.translation(+1, 0, 0), this.materials.exp);
        this.shapes.f4.draw(context, program_state, Mat4.translation(+3, 0, 0), this.materials.exp);
    }
}

class Shader_Version_1 extends Shader {
    constructor() {
        super();
    }

    shared_glsl_code() {
        // ********* SHARED CODE, INCLUDED IN BOTH SHADERS *********
        return ` 
        precision mediump float;
        `;
    }

    vertex_glsl_code() {
        // ********* VERTEX SHADER *********
        return this.shared_glsl_code() + `
            attribute vec3 position, normal;       
            
            uniform mat4 model_transform;
            uniform mat4 projection_camera_model_transform;
    
            void main(){                                                                   
                // The vertex's final resting place (in NDCS):
                gl_Position = projection_camera_model_transform * vec4( position, 1.0 ); // <---
            } `;
    }

    fragment_glsl_code() {
        // ********* FRAGMENT SHADER *********
        return this.shared_glsl_code() + `
            uniform vec4 shape_color; // <---
        
            void main(){                                                           
                gl_FragColor = shape_color;
            } `;
    }

    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        // update_GPU():  Defining how to synchronize our JavaScript's variables to the GPU's:
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform],
            PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false,
            Matrix.flatten_2D_to_1D(PCM.transposed()));

        // Set uniform parameters
        context.uniform4fv(gpu_addresses.shape_color, material.color);
    }
}

class Shader_Version_2 extends Shader {
    constructor() {
        super();
    }

    shared_glsl_code() {
        // ********* SHARED CODE, INCLUDED IN BOTH SHADERS *********
        return ` 
        precision mediump float;
        
        varying vec4 position_WCS; // <---
        `;
    }

    vertex_glsl_code() {
        // ********* VERTEX SHADER *********
        return this.shared_glsl_code() + `
            attribute vec3 position, normal;       
            
            uniform mat4 model_transform;
            uniform mat4 projection_camera_model_transform;
    
            void main(){                                                                   
                // The vertex's final resting place (in NDCS):
                gl_Position = projection_camera_model_transform * vec4( position, 1.0 ); 
                position_WCS = model_transform * vec4( position, 1.0 ); 
            } `;
    }

    fragment_glsl_code() {
        // ********* FRAGMENT SHADER *********
        return this.shared_glsl_code() + `
            uniform vec4 shape_color; 
            uniform vec4 second_color; // <---
        
            void main(){              
                float factor = 0.5 + 0.5 * sin(position_WCS.x * 5.0);
                vec4 mixed_color = factor * second_color + (1.0 - factor) * shape_color;
                gl_FragColor = mixed_color;
            } `;
    }

    // CHANGED
    send_material(gl, gpu, material) {
        // send_material(): Send the desired shape-wide material qualities to the
        // graphics card, where they will tweak the Phong lighting formula.
        gl.uniform4fv(gpu.shape_color, material.color);
        gl.uniform4fv(gpu.second_color, material.second_color); // <---
        gl.uniform1f(gpu.ambient, material.ambient);
        gl.uniform1f(gpu.diffusivity, material.diffusivity);
        gl.uniform1f(gpu.specularity, material.specularity);
        gl.uniform1f(gpu.smoothness, material.smoothness);
    }

    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        // update_GPU():  Defining how to synchronize our JavaScript's variables to the GPU's:
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform],
            PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false,
            Matrix.flatten_2D_to_1D(PCM.transposed()));

        // Set uniform parameters
        context.uniform4fv(gpu_addresses.shape_color, material.color);
        context.uniform4fv(gpu_addresses.second_color, material.second_color); // <---
    }
}

class Shader_Version_3 extends Shader {
    constructor() {
        super();
    }

    shared_glsl_code() {
        // ********* SHARED CODE, INCLUDED IN BOTH SHADERS *********
        return ` 
        precision mediump float;
        
        varying vec4 position_OCS; // <---
        `;
    }

    vertex_glsl_code() {
        // ********* VERTEX SHADER *********
        return this.shared_glsl_code() + `
            attribute vec3 position, normal;       
            
            uniform mat4 model_transform;
            uniform mat4 projection_camera_model_transform;
    
            void main(){                                                                   
                // The vertex's final resting place (in NDCS):
                gl_Position = projection_camera_model_transform * vec4( position, 1.0 ); 
                position_OCS = vec4( position, 1.0 ); // <---
            } `;
    }

    fragment_glsl_code() {
        // ********* FRAGMENT SHADER *********
        return this.shared_glsl_code() + `
            uniform vec4 shape_color; 
        
            void main(){              
                float factor = 0.5 + 0.5 * sin(position_OCS.x * 10.0 + position_OCS.y * 10.0);
                vec4 mixed_color =  vec4(shape_color.xyz, factor);
                gl_FragColor = mixed_color;
            } `;
    }

    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        // update_GPU():  Defining how to synchronize our JavaScript's variables to the GPU's:
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform],
            PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false,
            Matrix.flatten_2D_to_1D(PCM.transposed()));

        // Set uniform parameters
        context.uniform4fv(gpu_addresses.shape_color, material.color);
    }
}

class Example_Shader extends Shader {
    // This is a Shader using Phong_Shader as template

    constructor(num_lights = 2) {
        super();
        this.num_lights = num_lights;
    }

    shared_glsl_code() {
        // ********* SHARED CODE, INCLUDED IN BOTH SHADERS *********
        return ` 
        precision mediump float;
        const int N_LIGHTS = ` + this.num_lights + `;
        uniform float ambient, diffusivity, specularity, smoothness;
        uniform vec4 light_positions_or_vectors[N_LIGHTS], light_colors[N_LIGHTS];
        uniform float light_attenuation_factors[N_LIGHTS];
        uniform vec4 shape_color;
        uniform vec3 squared_scale, camera_center;
        // Specifier "varying" means a variable's final value will be passed from the vertex shader
        // on to the next phase (fragment shader), then interpolated per-fragment, weighted by the
        // pixel fragment's proximity to each of the 3 vertices (barycentric interpolation).
        
        // [Example]: We want to calculate color in VS <---
        //            So the varying variable should be color <---
        //            No need for object normal and position <---
        //varying vec3 N, vertex_worldspace;  // <---
        varying vec3 vertex_color;  // <---
        
        // ***** PHONG SHADING HAPPENS HERE: *****                                       
        vec3 phong_model_lights( vec3 N, vec3 vertex_worldspace ){                                        
            // phong_model_lights():  Add up the lights' contributions.
            vec3 E = normalize( camera_center - vertex_worldspace );
            vec3 result = vec3( 0.0 );
            for(int i = 0; i < N_LIGHTS; i++){
                // Lights store homogeneous coords - either a position or vector.  If w is 0, the 
                // light will appear directional (uniform direction from all points), and we 
                // simply obtain a vector towards the light by directly using the stored value.
                // Otherwise if w is 1 it will appear as a point light -- compute the vector to 
                // the point light's location from the current surface point.  In either case, 
                // fade (attenuate) the light as the vector needed to reach it gets longer.  
                vec3 surface_to_light_vector = light_positions_or_vectors[i].xyz - 
                                               light_positions_or_vectors[i].w * vertex_worldspace;                                             
                float distance_to_light = length( surface_to_light_vector );
                vec3 L = normalize( surface_to_light_vector );
                vec3 H = normalize( L + E );
                // Compute the diffuse and specular components from the Phong
                // Reflection Model, using Blinn's "halfway vector" method:
                float diffuse  =      max( dot( N, L ), 0.0 );
                float specular = pow( max( dot( N, H ), 0.0 ), smoothness );
                float attenuation = 1.0 / (1.0 + light_attenuation_factors[i] * distance_to_light * distance_to_light );
                
                vec3 light_contribution = shape_color.xyz * light_colors[i].xyz * diffusivity * diffuse
                                                          + light_colors[i].xyz * specularity * specular;
                result += attenuation * light_contribution;
            }
            return result;
        }                                  
        vec3 color_based_on_normal( vec3 N, vec3 vertex_worldspace ){ // <---                                       
            vec3 E = normalize( camera_center - vertex_worldspace );
            vec3 result = vec3( 0.0 );
            result += cross(N, E); 
            return result;
        } 
        `;
    }

    vertex_glsl_code() {
        // ********* VERTEX SHADER *********
        return this.shared_glsl_code() + `
            attribute vec3 position, normal;                            
            // Position is expressed in object coordinates.
            
            uniform mat4 model_transform;
            uniform mat4 projection_camera_model_transform;
    
            void main(){                                                                   
                // The vertex's final resting place (in NDCS):
                gl_Position = projection_camera_model_transform * vec4( position, 1.0 );
                // The final normal vector in screen space.
                vec3 normal_worldspace = normalize( mat3( model_transform ) * normal / squared_scale);  // <---
                vec3 vertex_worldspace = ( model_transform * vec4( position, 1.0 ) ).xyz;  // <---
                vertex_color = color_based_on_normal(normal_worldspace, vertex_worldspace);  // <---
            } `;
    }

    fragment_glsl_code() {
        // ********* FRAGMENT SHADER *********
        // A fragment is a pixel that's overlapped by the current triangle.
        // Fragments affect the final image or get discarded due to depth.
        return this.shared_glsl_code() + `
            void main(){                                                           
                // In FS, we just pass the interpolated color to the next stage.  // <---
                gl_FragColor = vec4( vertex_color.xyz, 1.0 );  // <---
            } `;
    }

    send_material(gl, gpu, material) {
        // send_material(): Send the desired shape-wide material qualities to the
        // graphics card, where they will tweak the Phong lighting formula.
        gl.uniform4fv(gpu.shape_color, material.color);
        gl.uniform1f(gpu.ambient, material.ambient);
        gl.uniform1f(gpu.diffusivity, material.diffusivity);
        gl.uniform1f(gpu.specularity, material.specularity);
        gl.uniform1f(gpu.smoothness, material.smoothness);
    }

    send_gpu_state(gl, gpu, gpu_state, model_transform) {
        // send_gpu_state():  Send the state of our whole drawing context to the GPU.
        const O = vec4(0, 0, 0, 1), camera_center = gpu_state.camera_transform.times(O).to3();
        gl.uniform3fv(gpu.camera_center, camera_center);
        // Use the squared scale trick from "Eric's blog" instead of inverse transpose matrix:
        const squared_scale = model_transform.reduce(
            (acc, r) => {
                return acc.plus(vec4(...r).times_pairwise(r))
            }, vec4(0, 0, 0, 0)).to3();
        gl.uniform3fv(gpu.squared_scale, squared_scale);
        // Send the current matrices to the shader.  Go ahead and pre-compute
        // the products we'll need of the of the three special matrices and just
        // cache and send those.  They will be the same throughout this draw
        // call, and thus across each instance of the vertex shader.
        // Transpose them since the GPU expects matrices as column-major arrays.
        const PCM = gpu_state.projection_transform.times(gpu_state.camera_inverse).times(model_transform);
        gl.uniformMatrix4fv(gpu.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        gl.uniformMatrix4fv(gpu.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));

        // Omitting lights will show only the material color, scaled by the ambient term:
        if (!gpu_state.lights.length)
            return;

        const light_positions_flattened = [], light_colors_flattened = [];
        for (let i = 0; i < 4 * gpu_state.lights.length; i++) {
            light_positions_flattened.push(gpu_state.lights[Math.floor(i / 4)].position[i % 4]);
            light_colors_flattened.push(gpu_state.lights[Math.floor(i / 4)].color[i % 4]);
        }
        gl.uniform4fv(gpu.light_positions_or_vectors, light_positions_flattened);
        gl.uniform4fv(gpu.light_colors, light_colors_flattened);
        gl.uniform1fv(gpu.light_attenuation_factors, gpu_state.lights.map(l => l.attenuation));
    }

    update_GPU(context, gpu_addresses, gpu_state, model_transform, material) {
        // update_GPU(): Define how to synchronize our JavaScript's variables to the GPU's.  This is where the shader
        // recieves ALL of its inputs.  Every value the GPU wants is divided into two categories:  Values that belong
        // to individual objects being drawn (which we call "Material") and values belonging to the whole scene or
        // program (which we call the "Program_State").  Send both a material and a program state to the shaders
        // within this function, one data field at a time, to fully initialize the shader for a draw.

        // Fill in any missing fields in the Material object with custom defaults for this shader:
        const defaults = {color: color(0, 0, 0, 1), ambient: 0, diffusivity: 1, specularity: 1, smoothness: 40};
        material = Object.assign({}, defaults, material);

        this.send_material(context, gpu_addresses, material);
        this.send_gpu_state(context, gpu_addresses, gpu_state, model_transform);
    }
}